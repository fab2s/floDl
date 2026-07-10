//! Prefetch pipeline internals for streaming mode.
//!
//! Not part of the public API. Used by [`DataLoader`](super::DataLoader)
//! when the dataset does not fit in VRAM.
//!
//! # Pipeline shape
//!
//! Single-stage (CPU targets, distributed `LoadBatch`): one worker
//! thread fetches from the dataset and forwards on the batch channel;
//! the channel itself is the read-ahead buffer.
//!
//! Two-stage (CUDA targets, `ring_slots > 0`): a reader thread fetches
//! batches from the dataset into a bounded pageable-RAM ring; the
//! worker thread drains the ring and runs the device transfer
//! (pin + async H2D + completion event). Storage-read latency then
//! overlaps the transfer stage's CPU work, raising the pipeline's
//! throughput ceiling from `1/(t_read + t_transfer)` to
//! `1/max(t_read, t_transfer)`, and the ring absorbs read jitter
//! (network storage). The ring bounds RAM in flight; the depth
//! governor independently bounds VRAM in flight.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::tensor::{Device, Result, Tensor};
use super::BatchDataSet;

// ---------------------------------------------------------------------------
// Depth governor
// ---------------------------------------------------------------------------

/// Shared depth-governor state for adaptive streaming prefetch.
///
/// Prefetch depth is a soft TARGET the worker respects (at most `target`
/// batches in flight, `sent - consumed`), not the channel capacity. That
/// makes depth adjustable at any moment, mid-epoch included: the sizing
/// policy (loader/consumer side) writes `target`, the worker reads it
/// before each fetch. The per-epoch channel keeps a generous fixed
/// capacity purely as a safety bound.
///
/// One instance lives on the `StreamingLoader` for the loader's lifetime,
/// shared with the worker thread and the live epoch iterator.
pub(crate) struct GovernorCtl {
    /// Soft cap on in-flight batches. Written by epoch sizing, the
    /// one-shot honest resize, `auto_resize`, and the worker's OOM
    /// halving. `0` is treated as `1` by the gate (a zero target would
    /// deadlock the pipeline).
    pub(crate) target: AtomicUsize,
    /// Batches pushed into the current epoch's channel (worker-side).
    pub(crate) sent: AtomicUsize,
    /// Batches drained from the current epoch's channel (consumer-side).
    pub(crate) consumed: AtomicUsize,
    /// Batches drained across the whole RUN. Drives the one-shot honest
    /// resize: once the consumer drains its second batch, the first
    /// batch's forward/backward/step have demonstrably executed, so a
    /// VRAM probe finally sees activations, gradients, and lazily
    /// created optimizer state.
    pub(crate) run_consumed: AtomicUsize,
    /// One-shot latch for the honest (post-first-step) resize.
    pub(crate) honest_resize_done: AtomicBool,
    /// Set by the epoch iterator's `Drop` when abandoned mid-epoch.
    /// Unblocks the worker's governor gate (consumed stops advancing on
    /// abandonment, so without this the gate could wait forever instead
    /// of reaching the failed `send` that ends the epoch).
    pub(crate) abandoned: AtomicBool,
}

impl GovernorCtl {
    pub(crate) fn new(initial_target: usize) -> Self {
        GovernorCtl {
            target: AtomicUsize::new(initial_target.max(1)),
            sent: AtomicUsize::new(0),
            consumed: AtomicUsize::new(0),
            run_consumed: AtomicUsize::new(0),
            honest_resize_done: AtomicBool::new(false),
            abandoned: AtomicBool::new(false),
        }
    }

    /// Reset per-epoch state and install the epoch's initial target.
    /// Run-level state (`run_consumed`, `honest_resize_done`) persists.
    pub(crate) fn begin_epoch(&self, target: usize) {
        self.sent.store(0, Ordering::Relaxed);
        self.consumed.store(0, Ordering::Relaxed);
        self.abandoned.store(false, Ordering::Relaxed);
        self.target.store(target.max(1), Ordering::Relaxed);
    }
}

/// Worker-side gate: wait until in-flight batches drop below the target.
/// Returns `false` when the epoch was abandoned (stop fetching).
fn governor_gate(gov: &GovernorCtl) -> bool {
    loop {
        if gov.abandoned.load(Ordering::Relaxed) {
            return false;
        }
        let sent = gov.sent.load(Ordering::Relaxed);
        let consumed = gov.consumed.load(Ordering::Relaxed);
        let target = gov.target.load(Ordering::Relaxed).max(1);
        if sent.saturating_sub(consumed) < target {
            return true;
        }
        // Idle wait; granularity is irrelevant next to batch times and
        // the worker has nothing else to do while the pipeline is full.
        thread::sleep(Duration::from_millis(1));
    }
}

/// CUDA-OOM retry budget: total patience is ~1s of consumer drain time.
pub(crate) const OOM_RETRY_ATTEMPTS: usize = 10;
pub(crate) const OOM_RETRY_SLEEP: Duration = Duration::from_millis(100);

/// Retry `attempt` while it fails with a CUDA-OOM error, calling
/// `on_oom` before each retry (empty-cache + target halving + sleep at
/// the call site). Prefetch OOM is usually transient: the consumer
/// frees batch tensors continuously, so the same allocation succeeds
/// after a short drain. Non-OOM errors (real dataset failures) return
/// immediately; exhausted retries return the last error.
pub(crate) fn retry_on_oom<T>(
    mut attempt: impl FnMut() -> Result<T>,
    mut on_oom: impl FnMut(),
) -> Result<T> {
    let mut result = attempt();
    for _ in 0..OOM_RETRY_ATTEMPTS {
        match &result {
            Err(e) if e.is_cuda_oom() => {
                on_oom();
                result = attempt();
            }
            _ => break,
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single prefetched batch, ready on the target device.
pub(crate) struct PrefetchedBatch {
    pub tensors: Vec<Tensor>,
    /// Event recorded after async H2D copy. Consumer waits on this.
    #[cfg(feature = "cuda")]
    pub ready_event: Option<crate::distributed::cuda_event::CudaEvent>,
}

/// Commands sent to the persistent worker thread.
pub(crate) enum WorkerCmd {
    /// Start a new epoch. Includes a fresh batch channel for this epoch.
    StartEpoch {
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        /// Per-epoch batch sender. Dropped when the epoch is done or cancelled.
        batch_tx: mpsc::SyncSender<Result<PrefetchedBatch>>,
        /// Depth governor shared with the loader and epoch iterator.
        governor: Arc<GovernorCtl>,
        /// Reader-ring capacity in batches. `0` = single-stage pipeline
        /// (fetch and transfer serialized on the worker thread); `> 0`
        /// spawns a reader thread that stages batches in pageable RAM.
        /// Sized by the loader from the host RAM budget.
        ring_slots: usize,
    },
    /// Open a distributed epoch: install the batch sender, then wait for
    /// `LoadBatch` commands. The channel stays open until the next
    /// `StartEpoch`/`StartDistributedEpoch`/`Stop`.
    StartDistributedEpoch {
        batch_tx: mpsc::SyncSender<Result<PrefetchedBatch>>,
    },
    /// Load a single batch (distributed mode). Worker sends the result on
    /// the channel from the preceding `StartDistributedEpoch`.
    LoadBatch {
        indices: Vec<usize>,
    },
    /// Shut down the worker.
    Stop,
}

// ---------------------------------------------------------------------------
// PrefetchWorker (persistent, lives for DataLoader lifetime)
// ---------------------------------------------------------------------------

/// Persistent background worker for streaming prefetch.
///
/// Created once at `DataLoader::build()`, lives until the DataLoader is
/// dropped. Keeps its dedicated CUDA stream alive across epochs.
///
/// Each epoch gets a fresh batch channel (created in `start_epoch()`),
/// so dropping an epoch iterator mid-epoch naturally cancels outstanding
/// work: the worker detects the closed channel and moves on.
pub(crate) struct PrefetchWorker {
    cmd_tx: mpsc::Sender<WorkerCmd>,
    handle: Option<JoinHandle<()>>,
    prefetch_depth: usize,
}

impl PrefetchWorker {
    /// Spawn the persistent worker thread.
    pub fn new(
        dataset: Arc<dyn BatchDataSet>,
        device: Device,
        prefetch_depth: usize,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCmd>();

        let handle = thread::spawn(move || {
            worker_loop(dataset, device, cmd_rx);
        });

        PrefetchWorker {
            cmd_tx,
            handle: Some(handle),
            prefetch_depth,
        }
    }

    /// Start a new epoch and return a receiver for the batches.
    ///
    /// `prefetch_depth` is the channel CAPACITY (safety ceiling for the
    /// epoch); the effective in-flight bound is the governor's target,
    /// adjustable mid-epoch. `ring_slots > 0` enables the two-stage
    /// pipeline (see module docs); `0` keeps fetch + transfer on one
    /// thread.
    pub fn start_epoch(
        &self,
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        governor: Arc<GovernorCtl>,
        ring_slots: usize,
    ) -> mpsc::Receiver<Result<PrefetchedBatch>> {
        let (batch_tx, batch_rx) =
            mpsc::sync_channel::<Result<PrefetchedBatch>>(self.prefetch_depth);

        let _ = self.cmd_tx.send(WorkerCmd::StartEpoch {
            indices,
            batch_size,
            drop_last,
            batch_tx,
            governor,
            ring_slots,
        });

        batch_rx
    }

    /// Open a distributed epoch: create one channel that persists across
    /// all batches. Follow with [`Self::load_batch()`] calls per batch.
    pub fn start_distributed_epoch(&self) -> mpsc::Receiver<Result<PrefetchedBatch>> {
        let (batch_tx, batch_rx) =
            mpsc::sync_channel::<Result<PrefetchedBatch>>(self.prefetch_depth);

        let _ = self.cmd_tx.send(WorkerCmd::StartDistributedEpoch { batch_tx });

        batch_rx
    }

    /// Send a single batch of indices for loading (distributed mode).
    /// The result arrives on the receiver from [`Self::start_distributed_epoch()`].
    pub fn load_batch(&self, indices: Vec<usize>) {
        let _ = self.cmd_tx.send(WorkerCmd::LoadBatch { indices });
    }

    /// Current prefetch depth (channel capacity for next epoch).
    pub fn prefetch_depth(&self) -> usize {
        self.prefetch_depth
    }

    /// Update prefetch depth. Takes effect on the next epoch (the channel
    /// is recreated with the new capacity in `start_epoch()`).
    pub fn set_prefetch_depth(&mut self, depth: usize) {
        self.prefetch_depth = depth;
    }
}

impl Drop for PrefetchWorker {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(WorkerCmd::Stop);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

fn worker_loop(
    dataset: Arc<dyn BatchDataSet>,
    device: Device,
    cmd_rx: mpsc::Receiver<WorkerCmd>,
) {
    // Create a dedicated CUDA stream for H2D transfers (lives across epochs).
    #[cfg(feature = "cuda")]
    let copy_stream = if device.is_cuda() {
        crate::distributed::cuda_stream::CudaStream::new(device, false).ok()
    } else {
        None
    };

    // Distributed epoch channel, kept alive across LoadBatch commands.
    let mut dist_tx: Option<mpsc::SyncSender<Result<PrefetchedBatch>>> = None;

    for cmd in &cmd_rx {
        match cmd {
            WorkerCmd::StartEpoch {
                indices,
                batch_size,
                drop_last,
                batch_tx,
                governor,
                ring_slots,
            } => {
                dist_tx = None; // close any distributed channel

                if ring_slots > 0 {
                    run_two_stage_epoch(
                        &dataset,
                        device,
                        indices,
                        batch_size,
                        drop_last,
                        &batch_tx,
                        &governor,
                        ring_slots,
                        #[cfg(feature = "cuda")]
                        copy_stream.as_ref(),
                    );
                } else {
                    run_single_stage_epoch(
                        &dataset,
                        device,
                        &indices,
                        batch_size,
                        drop_last,
                        &batch_tx,
                        &governor,
                        #[cfg(feature = "cuda")]
                        copy_stream.as_ref(),
                    );
                }
                // batch_tx is dropped here, closing the epoch's channel.
            }
            WorkerCmd::StartDistributedEpoch { batch_tx } => {
                dist_tx = Some(batch_tx);
            }
            WorkerCmd::LoadBatch { indices } => {
                if let Some(ref tx) = dist_tx {
                    // Same transient-OOM patience as the epoch path; the
                    // coordinator paces in-flight batches here, so there
                    // is no governor target to shrink.
                    let result = if device.is_cuda() {
                        retry_on_oom(
                            || {
                                fetch_and_transfer(
                                    &*dataset,
                                    &indices,
                                    device,
                                    #[cfg(feature = "cuda")]
                                    copy_stream.as_ref(),
                                )
                            },
                            || {
                                crate::tensor::cuda_empty_cache();
                                thread::sleep(OOM_RETRY_SLEEP);
                            },
                        )
                    } else {
                        fetch_and_transfer(
                            &*dataset,
                            &indices,
                            device,
                            #[cfg(feature = "cuda")]
                            copy_stream.as_ref(),
                        )
                    };
                    if tx.send(result).is_err() {
                        dist_tx = None; // consumer dropped
                    }
                }
            }
            WorkerCmd::Stop => break,
        }
    }
}

/// Single-stage epoch: fetch and transfer serialized on the worker
/// thread. Used when `ring_slots == 0` (CPU targets, where the batch
/// channel itself is the read-ahead buffer and there is no transfer
/// stage to overlap; or a RAM budget too tight for a reader ring).
#[allow(clippy::too_many_arguments)]
fn run_single_stage_epoch(
    dataset: &Arc<dyn BatchDataSet>,
    device: Device,
    indices: &[usize],
    batch_size: usize,
    drop_last: bool,
    batch_tx: &mpsc::SyncSender<Result<PrefetchedBatch>>,
    governor: &GovernorCtl,
    #[cfg(feature = "cuda")] copy_stream: Option<&crate::distributed::cuda_stream::CudaStream>,
) {
    let n = indices.len();
    let mut start = 0;

    while start < n {
        let end = (start + batch_size).min(n);
        if drop_last && (end - start) < batch_size {
            break;
        }

        // Governor gate: at most `target` batches in flight.
        if !governor_gate(governor) {
            break; // epoch abandoned
        }

        let batch_indices = &indices[start..end];
        start = end;

        // Transient prefetch OOM (consumer will drain): free
        // the allocator cache, halve the in-flight target so
        // the overcommit self-heals instead of re-OOMing on
        // every batch, and give the consumer time to drain.
        let result = if device.is_cuda() {
            retry_on_oom(
                || {
                    fetch_and_transfer(
                        &**dataset,
                        batch_indices,
                        device,
                        #[cfg(feature = "cuda")]
                        copy_stream,
                    )
                },
                || oom_backoff(governor),
            )
        } else {
            fetch_and_transfer(
                &**dataset,
                batch_indices,
                device,
                #[cfg(feature = "cuda")]
                copy_stream,
            )
        };

        // If the consumer dropped (epoch iterator dropped mid-epoch),
        // the send fails. We stop this epoch and wait for the next command.
        if batch_tx.send(result).is_err() {
            break;
        }
        governor.sent.fetch_add(1, Ordering::Relaxed);
    }
}

/// Two-stage epoch: a reader thread stages CPU-side batches in a
/// bounded pageable ring; this thread drains the ring and runs the
/// device transfer. See the module docs for the cost model.
#[allow(clippy::too_many_arguments)]
fn run_two_stage_epoch(
    dataset: &Arc<dyn BatchDataSet>,
    device: Device,
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    batch_tx: &mpsc::SyncSender<Result<PrefetchedBatch>>,
    governor: &GovernorCtl,
    ring_slots: usize,
    #[cfg(feature = "cuda")] copy_stream: Option<&crate::distributed::cuda_stream::CudaStream>,
) {
    let (ring_tx, ring_rx) = mpsc::sync_channel::<Result<Vec<Tensor>>>(ring_slots);
    let reader_dataset = Arc::clone(dataset);
    let reader = thread::spawn(move || {
        reader_loop(reader_dataset, indices, batch_size, drop_last, ring_tx);
    });

    loop {
        // Governor gate first: while the VRAM pipeline is full, ready
        // batches wait in the pageable ring, not on the device.
        if !governor_gate(governor) {
            break; // epoch abandoned
        }
        let cpu_batch = match ring_rx.recv() {
            Ok(b) => b,
            Err(_) => break, // reader done: epoch exhausted
        };

        // Same transient-OOM patience as the single-stage path; only
        // the transfer half retries (the batch is already in RAM).
        let result = match cpu_batch {
            Ok(tensors) => {
                if device.is_cuda() {
                    retry_on_oom(
                        || {
                            transfer_batch(
                                &tensors,
                                device,
                                #[cfg(feature = "cuda")]
                                copy_stream,
                            )
                        },
                        || oom_backoff(governor),
                    )
                } else {
                    transfer_batch(
                        &tensors,
                        device,
                        #[cfg(feature = "cuda")]
                        copy_stream,
                    )
                }
            }
            Err(e) => Err(e),
        };

        if batch_tx.send(result).is_err() {
            break; // consumer dropped mid-epoch
        }
        governor.sent.fetch_add(1, Ordering::Relaxed);
    }

    // Dropping the ring receiver fails the reader's next send, so an
    // early break (abandoned epoch, consumer gone) unwinds the reader
    // too. Join before returning: the worker must not process the next
    // command while a stale reader still holds the dataset.
    drop(ring_rx);
    let _ = reader.join();
}

/// Reader stage: fetch batches from the dataset in index order and push
/// them into the ring. Pure dataset I/O — no pinning, no device work
/// (the transfer stage pins, so the ring holds pageable RAM and the
/// I/O wait here genuinely overlaps the transfer stage's CPU work).
fn reader_loop(
    dataset: Arc<dyn BatchDataSet>,
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    ring_tx: mpsc::SyncSender<Result<Vec<Tensor>>>,
) {
    let n = indices.len();
    let mut start = 0;

    while start < n {
        let end = (start + batch_size).min(n);
        if drop_last && (end - start) < batch_size {
            break;
        }

        let result = dataset.get_batch(&indices[start..end]);
        start = end;

        // Errors travel the ring like batches (the consumer surfaces
        // them); a failed send means the transfer stage is gone.
        if ring_tx.send(result).is_err() {
            break;
        }
    }
}

/// Shared OOM back-off for both pipeline shapes: halve the in-flight
/// target so the overcommit self-heals instead of re-OOMing on every
/// batch, free the allocator cache, give the consumer time to drain.
fn oom_backoff(governor: &GovernorCtl) {
    let t = governor.target.load(Ordering::Relaxed);
    governor.target.store((t / 2).max(1), Ordering::Relaxed);
    crate::tensor::cuda_empty_cache();
    thread::sleep(OOM_RETRY_SLEEP);
}

/// Fetch a batch from the dataset and transfer to the target device.
/// Single-stage path: both halves serialized on the calling thread.
fn fetch_and_transfer(
    dataset: &dyn BatchDataSet,
    indices: &[usize],
    device: Device,
    #[cfg(feature = "cuda")] copy_stream: Option<&crate::distributed::cuda_stream::CudaStream>,
) -> Result<PrefetchedBatch> {
    let tensors = dataset.get_batch(indices)?;
    transfer_batch(
        &tensors,
        device,
        #[cfg(feature = "cuda")]
        copy_stream,
    )
}

/// Transfer stage: move CPU-side batch tensors to the target device
/// (pin + async H2D on the copy stream, completion event recorded).
/// On a CPU target this is a pass-through (shallow clones: refcount
/// bumps on shared storage, no data copy).
fn transfer_batch(
    tensors: &[Tensor],
    device: Device,
    #[cfg(feature = "cuda")] copy_stream: Option<&crate::distributed::cuda_stream::CudaStream>,
) -> Result<PrefetchedBatch> {
    if !device.is_cuda() {
        return Ok(PrefetchedBatch {
            tensors: tensors.to_vec(),
            #[cfg(feature = "cuda")]
            ready_event: None,
        });
    }

    // Pin memory and async-copy to GPU on dedicated stream
    #[cfg(feature = "cuda")]
    {
        use crate::distributed::cuda_event::{CudaEvent, CudaEventFlags};
        use crate::distributed::cuda_stream::StreamGuard;

        let mut on_device = Vec::with_capacity(tensors.len());

        if let Some(stream) = copy_stream {
            let _guard = StreamGuard::new(stream);
            for t in tensors {
                let pinned = t.pin_memory()?;
                on_device.push(pinned.to_device_async(device)?);
            }

            // Record completion event on the copy stream
            let event = CudaEvent::new(CudaEventFlags::DisableTiming)?;
            event.record_on(stream)?;

            return Ok(PrefetchedBatch {
                tensors: on_device,
                ready_event: Some(event),
            });
        }

        // Fallback: synchronous transfer (no stream available)
        for t in tensors {
            let pinned = t.pin_memory()?;
            on_device.push(pinned.to_device(device)?);
        }

        Ok(PrefetchedBatch {
            tensors: on_device,
            ready_event: None,
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Without CUDA feature, just return CPU tensors
        Ok(PrefetchedBatch {
            tensors: tensors.to_vec(),
        })
    }
}
