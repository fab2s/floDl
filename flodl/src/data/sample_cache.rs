//! Read-through, sample-keyed RAM cache for streaming datasets.
//!
//! Not part of the public API. Sits behind [`DataSet::get`](super::DataSet::get)
//! (inside `DataSetAdapter`), below batching: batches are not reusable
//! across epochs (a reshuffle changes their composition), samples are.
//! Epoch 1 populates while training; later epochs read at RAM speed
//! instead of storage speed. Staged content is reshuffle-invariant by
//! construction: the cache is keyed by sample identity, and a reshuffle
//! changes only the order function, never the content set.
//!
//! # Admission policy: fill until full, evict nothing
//!
//! Each epoch touches every sample exactly once in a fresh random
//! order, so for a cache holding K of N samples the expected hit rate
//! is K/N for ANY eviction policy — no choice of which K to keep beats
//! any other against a uniformly reshuffled scan. Admit-until-full
//! delivers that same K/N with zero churn (no eviction traffic, no
//! write contention after warm-up). Smarter eviction only becomes
//! meaningful when a disk tier exists below (spill-vs-drop rather than
//! keep-vs-drop).
//!
//! # Concurrency
//!
//! Lock-free by construction: reads are one atomic load per lookup
//! (`OnceLock::get`), writes are per-slot one-time. No global lock
//! anywhere near the data path. The byte counter is advisory under
//! concurrent writers (two racing inserts can overshoot the budget by
//! less than one sample each); in practice a single reader thread
//! populates it.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::tensor::{Result, Tensor, TensorError};

/// Sample-keyed read-through cache. One instance per `from_dataset`
/// loader, shared between the adapter (inside the `dyn BatchDataSet`
/// box) and the loader (which refreshes the budget at each `epoch()`).
///
/// Dormant until a budget is installed: with `budget == 0` a miss is a
/// pure pass-through (two atomic loads of overhead) and nothing is
/// retained.
pub(crate) struct SampleCache {
    /// One slot per sample index, set at most once.
    slots: Vec<OnceLock<Vec<Tensor>>>,
    /// Bytes currently retained (sum of cached samples' nbytes).
    bytes: AtomicUsize,
    /// Admission ceiling in bytes. Includes already-retained bytes:
    /// the per-epoch refresh computes `bytes() + headroom`, so a
    /// shrinking headroom stops NEW admissions but never drops staged
    /// content ("keep the subset already there").
    budget: AtomicUsize,
    /// Optional local-disk tier: admits what RAM declined, serves
    /// misses at local-disk speed instead of source speed. Attached at
    /// `build()` when `disk_stage(gb)` is set.
    disk: OnceLock<DiskStage>,
}

impl SampleCache {
    pub(crate) fn new(n: usize) -> Self {
        let mut slots = Vec::with_capacity(n);
        slots.resize_with(n, OnceLock::new);
        SampleCache {
            slots,
            bytes: AtomicUsize::new(0),
            budget: AtomicUsize::new(0),
            disk: OnceLock::new(),
        }
    }

    /// Attach the local-disk tier (once, at `build()`).
    pub(crate) fn attach_disk(&self, stage: DiskStage) {
        let _ = self.disk.set(stage);
    }

    #[cfg(test)]
    pub(crate) fn disk(&self) -> Option<&DiskStage> {
        self.disk.get()
    }

    /// Bytes currently retained.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Number of samples currently cached (test/diagnostic).
    #[cfg(test)]
    pub(crate) fn cached_count(&self) -> usize {
        self.slots.iter().filter(|s| s.get().is_some()).count()
    }

    /// Install the admission ceiling. Called once per `epoch()` by the
    /// streaming loader; `0` means no new admissions (retained content
    /// stays served).
    pub(crate) fn set_budget(&self, bytes: usize) {
        self.budget.store(bytes, Ordering::Relaxed);
    }

    /// Read-through lookup, cascading RAM → disk → source: return the
    /// cached sample (shallow tensor clones — refcount bumps, no data
    /// copy), else read it from the disk tier, else run `fetch` and
    /// admit the result (RAM while the budget allows, disk when RAM
    /// declined).
    ///
    /// Cached tensors share storage with every clone handed out; the
    /// adapter's stacking path only reads them (`copy_` out of the
    /// sample into the batch row), never mutates. Datasets returning
    /// views into their own buffers cache the view (no byte
    /// duplication; `nbytes` accounting is conservative).
    ///
    /// A disk-tier READ error propagates (the pack file is ours; a
    /// failed read is a real disk problem the user must see). Disk
    /// WRITE errors never fail training — the sample is already in
    /// hand — they latch the stage off, loudly, and the run continues
    /// source-backed.
    pub(crate) fn get_or_fetch(
        &self,
        index: usize,
        fetch: impl FnOnce() -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        if let Some(staged) = self.lookup(index) {
            return staged;
        }
        let sample = fetch()?;
        self.admit(index, &sample);
        Ok(sample)
    }

    /// Tier lookup half of the read-through: RAM hit (shallow clones),
    /// else disk read. `None` = not staged anywhere, caller fetches
    /// from the source.
    pub(crate) fn lookup(&self, index: usize) -> Option<Result<Vec<Tensor>>> {
        if let Some(hit) = self.slots.get(index).and_then(|s| s.get()) {
            return Some(Ok(hit.clone()));
        }
        self.disk.get().and_then(|stage| stage.read(index))
    }

    /// Admission half of the read-through: RAM while the budget allows,
    /// overflow to the local-disk tier when RAM declines (at most once
    /// per sample), so later reads hit disk speed instead of source
    /// speed.
    pub(crate) fn admit(&self, index: usize, sample: &[Tensor]) {
        let mut ram_admitted = false;
        if let Some(slot) = self.slots.get(index) {
            let sample_bytes: usize = sample.iter().map(|t| t.nbytes()).sum();
            if self.bytes.load(Ordering::Relaxed) + sample_bytes
                <= self.budget.load(Ordering::Relaxed)
                && slot.set(sample.to_vec()).is_ok()
            {
                self.bytes.fetch_add(sample_bytes, Ordering::Relaxed);
                ram_admitted = true;
            }
        }
        if !ram_admitted {
            if let Some(stage) = self.disk.get() {
                stage.admit(index, sample);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DiskStage (local-drive overflow tier)
// ---------------------------------------------------------------------------

/// Sequence for unique pack-file names when one process builds several
/// staged loaders.
static STAGE_SEQ: AtomicUsize = AtomicUsize::new(0);

/// Local-disk overflow tier below the RAM sample cache.
///
/// One append-only pack file (not one file per sample: sequential
/// append is every drive's fast path, and millions of small files are
/// inode pressure), plus an in-RAM offset index with the same
/// set-once-slot pattern as the RAM tier. Admission is the same
/// fill-until-full-evict-nothing policy, one tier down: under a
/// uniformly reshuffled scan, WHICH samples sit in which tier does not
/// change the hit profile, so there is nothing for a smarter placement
/// to win until access becomes non-uniform (reservation-constrained
/// windows).
///
/// Ephemeral: the pack file is removed on drop. A persistent
/// cross-run stage needs dataset identity and invalidation (manifest
/// work) and belongs to the host-scoped staging layer.
pub(crate) struct DiskStage {
    /// Read handle: positioned reads (`read_exact_at`), no shared seek
    /// state, so reads stay lock-free alongside appends.
    #[cfg(unix)]
    reader: File,
    /// Append state: file cursor is always at the end.
    writer: Mutex<DiskWriter>,
    /// `(offset, len)` per sample index, set once after a COMPLETED
    /// write (readers never see a partially written entry).
    offsets: Vec<OnceLock<(u64, u64)>>,
    /// Admission ceiling in bytes.
    budget: u64,
    /// Latched on the first write failure: stop admitting, keep
    /// serving what was staged. Never fails training.
    failed: AtomicBool,
    /// Pack-file path, removed on drop.
    path: PathBuf,
}

struct DiskWriter {
    file: File,
    offset: u64,
}

impl DiskStage {
    /// Create the pack file under `dir` (created if missing). Errors
    /// are loud: `disk_stage` is an explicit knob, an unusable
    /// directory must not degrade silently.
    pub(crate) fn create(dir: &Path, budget_bytes: u64, n: usize) -> Result<Self> {
        std::fs::create_dir_all(dir).map_err(|e| {
            TensorError::new(&format!(
                "DataLoader: disk_stage directory {} cannot be created: {e}",
                dir.display()
            ))
        })?;
        warn_if_ram_backed(dir);

        let seq = STAGE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("flodl-stage-{}-{seq}.pack", std::process::id()));
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                TensorError::new(&format!(
                    "DataLoader: disk_stage pack file {} cannot be created: {e}",
                    path.display()
                ))
            })?;
        #[cfg(unix)]
        let reader = File::open(&path).map_err(|e| {
            TensorError::new(&format!(
                "DataLoader: disk_stage pack file {} cannot be reopened: {e}",
                path.display()
            ))
        })?;

        let mut offsets = Vec::with_capacity(n);
        offsets.resize_with(n, OnceLock::new);

        Ok(DiskStage {
            #[cfg(unix)]
            reader,
            writer: Mutex::new(DiskWriter { file, offset: 0 }),
            offsets,
            budget: budget_bytes,
            failed: AtomicBool::new(false),
            path,
        })
    }

    /// Bytes currently staged (test/diagnostic).
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> u64 {
        self.writer.lock().map(|w| w.offset).unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn staged_count(&self) -> usize {
        self.offsets.iter().filter(|s| s.get().is_some()).count()
    }

    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Stage a sample if the budget allows. Best-effort: encode or
    /// write failures latch the stage off loudly instead of failing
    /// training (the caller already holds the sample).
    pub(crate) fn admit(&self, index: usize, sample: &[Tensor]) {
        if self.failed.load(Ordering::Relaxed) {
            return;
        }
        let Some(slot) = self.offsets.get(index) else {
            return;
        };
        if slot.get().is_some() {
            return;
        }

        let encoded = match encode_sample(sample) {
            Ok(b) => b,
            Err(e) => {
                self.fail(&format!("sample encode failed: {e}"));
                return;
            }
        };
        let len = encoded.len() as u64;

        let offset = {
            let mut w = match self.writer.lock() {
                Ok(w) => w,
                Err(_) => return, // poisoned by a panicking writer: stand down
            };
            if w.offset.saturating_add(len) > self.budget {
                return; // budget full: decline, not a failure
            }
            let offset = w.offset;
            if let Err(e) = w.file.write_all(&encoded) {
                drop(w);
                self.fail(&format!("pack-file write failed: {e}"));
                return;
            }
            w.offset += len;
            offset
        };

        let _ = slot.set((offset, len));
    }

    /// Read a staged sample. `None` = not staged (caller falls through
    /// to the source); `Some(Err)` = the pack file is damaged or
    /// unreadable, a real disk problem that must surface.
    pub(crate) fn read(&self, index: usize) -> Option<Result<Vec<Tensor>>> {
        let &(offset, len) = self.offsets.get(index)?.get()?;
        let mut buf = vec![0u8; len as usize];

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            if let Err(e) = self.reader.read_exact_at(&mut buf, offset) {
                return Some(Err(TensorError::new(&format!(
                    "DataLoader: disk_stage read failed at {}: {e}",
                    self.path.display()
                ))));
            }
        }
        #[cfg(not(unix))]
        {
            use std::io::{Seek, SeekFrom};
            let mut w = match self.writer.lock() {
                Ok(w) => w,
                Err(_) => {
                    return Some(Err(TensorError::new(
                        "DataLoader: disk_stage writer poisoned",
                    )))
                }
            };
            let end = w.offset;
            let read = w
                .file
                .seek(SeekFrom::Start(offset))
                .and_then(|_| w.file.read_exact(&mut buf))
                .and_then(|_| w.file.seek(SeekFrom::Start(end)).map(|_| ()));
            if let Err(e) = read {
                return Some(Err(TensorError::new(&format!(
                    "DataLoader: disk_stage read failed at {}: {e}",
                    self.path.display()
                ))));
            }
        }

        Some(decode_sample(&buf))
    }

    fn fail(&self, why: &str) {
        if !self.failed.swap(true, Ordering::Relaxed) {
            eprintln!(
                "flodl data: disk_stage disabled ({why}); training continues source-backed, \
                 already-staged samples keep serving"
            );
        }
    }
}

impl Drop for DiskStage {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Sample codec: tensor count + the checkpoint module's per-tensor
/// layout (ndim + shape + dtype tag + raw bytes) — one format, not two.
fn encode_sample(tensors: &[Tensor]) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        crate::nn::checkpoint::write_tensor_data(&mut buf, t)?;
    }
    Ok(buf)
}

fn decode_sample(bytes: &[u8]) -> Result<Vec<Tensor>> {
    let mut r = std::io::Cursor::new(bytes);
    let mut n4 = [0u8; 4];
    r.read_exact(&mut n4)
        .map_err(|e| TensorError::new(&format!("disk_stage: corrupt sample header: {e}")))?;
    let n = u32::from_le_bytes(n4) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(crate::nn::checkpoint::read_tensor_data(&mut r)?);
    }
    Ok(out)
}

/// Loud heads-up when the stage directory is RAM-backed (`/tmp` is
/// frequently tmpfs on Linux): the stage still works, but it spends
/// RAM, which defeats its purpose next to the byte-budgeted RAM tier.
fn warn_if_ram_backed(dir: &Path) {
    let Ok(mounts) = std::fs::read_to_string("/proc/mounts") else {
        return; // non-Linux: nothing to check
    };
    let target = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    if let Some(fstype) = fs_type_for(&target, &mounts) {
        if fstype == "tmpfs" || fstype == "ramfs" {
            eprintln!(
                "flodl data: disk_stage directory {} is on {fstype} (RAM-backed): the stage \
                 will spend RAM, not disk. Point .disk_stage_dir() at a real drive.",
                target.display()
            );
        }
    }
}

/// Filesystem type of the longest mount point containing `path`
/// (/proc/mounts format: device mountpoint fstype options ...).
fn fs_type_for(path: &Path, mounts: &str) -> Option<String> {
    let mut best: Option<(&str, &str)> = None;
    for line in mounts.lines() {
        let mut fields = line.split_whitespace();
        let (Some(_dev), Some(mount_point), Some(fstype)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if path.starts_with(mount_point)
            && best.is_none_or(|(b, _)| mount_point.len() > b.len())
        {
            best = Some((mount_point, fstype));
        }
    }
    best.map(|(_, t)| t.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn sample(value: f32) -> Vec<Tensor> {
        // 4 bytes per sample (one f32).
        vec![Tensor::from_f32(&[value], &[1], Device::CPU).unwrap()]
    }

    #[test]
    fn dormant_cache_is_pure_pass_through() {
        let cache = SampleCache::new(4);
        let mut fetches = 0;
        for _ in 0..3 {
            let s = cache
                .get_or_fetch(1, || {
                    fetches += 1;
                    Ok(sample(1.0))
                })
                .unwrap();
            assert_eq!(s[0].to_f64_vec().unwrap(), vec![1.0]);
        }
        assert_eq!(fetches, 3, "budget 0: nothing retained, every call fetches");
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.cached_count(), 0);
    }

    #[test]
    fn admits_until_budget_then_serves_hits() {
        let cache = SampleCache::new(4);
        cache.set_budget(8); // room for exactly two 4-byte samples

        let mut fetches = 0;
        for idx in 0..4 {
            let _ = cache
                .get_or_fetch(idx, || {
                    fetches += 1;
                    Ok(sample(idx as f32))
                })
                .unwrap();
        }
        assert_eq!(fetches, 4, "first pass: all misses");
        assert_eq!(cache.cached_count(), 2, "admission stopped at the budget");
        assert_eq!(cache.bytes(), 8);

        // Second pass: cached indices hit, the rest re-fetch.
        for idx in 0..4 {
            let s = cache
                .get_or_fetch(idx, || {
                    fetches += 1;
                    Ok(sample(idx as f32))
                })
                .unwrap();
            assert_eq!(s[0].to_f64_vec().unwrap(), vec![idx as f64]);
        }
        assert_eq!(fetches, 6, "two hits, two re-fetches");
    }

    #[test]
    fn budget_shrink_keeps_retained_content() {
        let cache = SampleCache::new(2);
        cache.set_budget(64);
        let _ = cache.get_or_fetch(0, || Ok(sample(7.0))).unwrap();
        assert_eq!(cache.cached_count(), 1);

        // Shrinking the budget below current bytes stops NEW
        // admissions but keeps serving what is already staged.
        cache.set_budget(0);
        let mut fetched = false;
        let s = cache
            .get_or_fetch(0, || {
                fetched = true;
                Ok(sample(0.0))
            })
            .unwrap();
        assert!(!fetched, "still a hit after budget shrink");
        assert_eq!(s[0].to_f64_vec().unwrap(), vec![7.0]);

        let _ = cache.get_or_fetch(1, || Ok(sample(1.0))).unwrap();
        assert_eq!(cache.cached_count(), 1, "no new admission at budget 0");
    }

    #[test]
    fn fetch_errors_pass_through_and_cache_nothing() {
        let cache = SampleCache::new(2);
        cache.set_budget(64);
        let out = cache.get_or_fetch(0, || {
            Err(crate::tensor::TensorError::new("io failure"))
        });
        assert!(out.is_err());
        assert_eq!(cache.cached_count(), 0);

        // A later successful fetch still admits.
        let _ = cache.get_or_fetch(0, || Ok(sample(1.0))).unwrap();
        assert_eq!(cache.cached_count(), 1);
    }

    #[test]
    fn out_of_range_index_is_fetch_only() {
        let cache = SampleCache::new(1);
        cache.set_budget(64);
        let s = cache.get_or_fetch(9, || Ok(sample(3.0))).unwrap();
        assert_eq!(s[0].to_f64_vec().unwrap(), vec![3.0]);
        assert_eq!(cache.bytes(), 0);
    }

    fn stage_dir() -> std::path::PathBuf {
        std::env::temp_dir().join("flodl-stage-tests")
    }

    /// Mixed-dtype sample: one f32 tensor + one i64 tensor.
    fn mixed_sample(v: f32, i: i64) -> Vec<Tensor> {
        vec![
            Tensor::from_f32(&[v, v + 1.0], &[2], Device::CPU).unwrap(),
            Tensor::from_i64(&[i], &[1], Device::CPU).unwrap(),
        ]
    }

    #[test]
    fn disk_stage_round_trips_samples_and_cleans_up() {
        let stage = DiskStage::create(&stage_dir(), 1 << 20, 4).unwrap();
        let path = stage.path().to_path_buf();
        assert!(path.exists());

        stage.admit(0, &mixed_sample(1.0, 10));
        stage.admit(2, &mixed_sample(3.0, 30));
        assert_eq!(stage.staged_count(), 2);
        assert!(stage.bytes() > 0);

        let s0 = stage.read(0).unwrap().unwrap();
        assert_eq!(s0[0].to_f64_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(s0[1].to_i64_vec().unwrap(), vec![10]);
        let s2 = stage.read(2).unwrap().unwrap();
        assert_eq!(s2[0].to_f64_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(s2[1].to_i64_vec().unwrap(), vec![30]);

        // Not staged / out of range: caller falls through to source.
        assert!(stage.read(1).is_none());
        assert!(stage.read(9).is_none());

        // Re-admitting an index is a no-op (set-once).
        let before = stage.bytes();
        stage.admit(0, &mixed_sample(9.0, 99));
        assert_eq!(stage.bytes(), before);

        drop(stage);
        assert!(!path.exists(), "pack file removed on drop");
    }

    #[test]
    fn disk_stage_budget_declines_without_failing() {
        let stage = DiskStage::create(&stage_dir(), 8, 4).unwrap();
        // A mixed sample encodes far past 8 bytes: declined, not failed.
        stage.admit(0, &mixed_sample(1.0, 1));
        assert_eq!(stage.staged_count(), 0);
        assert!(stage.read(0).is_none());
        // Decline is not the failure latch: admission stays open for
        // anything that would fit (nothing here, but the flag matters).
        assert!(!stage.failed.load(Ordering::Relaxed));
    }

    #[test]
    fn cache_cascades_ram_then_disk_then_source() {
        // RAM budget fits exactly one 4-byte sample; the disk tier
        // catches the rest.
        let cache = SampleCache::new(3);
        cache.set_budget(4);
        cache.attach_disk(DiskStage::create(&stage_dir(), 1 << 20, 3).unwrap());

        let mut fetches = 0;
        for idx in 0..3 {
            let s = cache
                .get_or_fetch(idx, || {
                    fetches += 1;
                    Ok(sample(idx as f32))
                })
                .unwrap();
            assert_eq!(s[0].to_f64_vec().unwrap(), vec![idx as f64]);
        }
        assert_eq!(fetches, 3, "first pass: all misses");
        assert_eq!(cache.cached_count(), 1, "RAM took one");
        assert_eq!(cache.disk().unwrap().staged_count(), 2, "disk took the rest");

        // Second pass: every index is served without touching the
        // source, from whichever tier holds it.
        for idx in 0..3 {
            let s = cache
                .get_or_fetch(idx, || {
                    fetches += 1;
                    Ok(sample(-1.0))
                })
                .unwrap();
            assert_eq!(s[0].to_f64_vec().unwrap(), vec![idx as f64]);
        }
        assert_eq!(fetches, 3, "second pass: zero source fetches");
    }

    #[test]
    fn fs_type_longest_mount_prefix_wins() {
        let mounts = "\
/dev/nvme0n1p2 / ext4 rw,relatime 0 0
tmpfs /tmp tmpfs rw,nosuid,nodev 0 0
/dev/sda1 /data ssdfs rw 0 0
tmpfs /data/scratch tmpfs rw 0 0
";
        let t = |p: &str| fs_type_for(Path::new(p), mounts);
        assert_eq!(t("/tmp/flodl").as_deref(), Some("tmpfs"));
        assert_eq!(t("/data/set").as_deref(), Some("ssdfs"));
        assert_eq!(t("/data/scratch/x").as_deref(), Some("tmpfs"));
        assert_eq!(t("/home/u").as_deref(), Some("ext4"));
    }
}
