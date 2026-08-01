//! Host RAM detection.

/// Host RAM snapshot: total and currently-available bytes.
///
/// `available_bytes` is the kernel's `MemAvailable` estimate: memory a
/// new workload can take without pushing the system into swap,
/// reclaimable page cache included. That makes it the honest baseline
/// for sizing staging buffers, since it already accounts for every other
/// process on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemInfo {
    /// Total physical RAM in bytes (`MemTotal`).
    pub total_bytes: u64,
    /// Kernel estimate of allocatable RAM in bytes (`MemAvailable`).
    pub available_bytes: u64,
}

impl MemInfo {
    /// Total physical RAM in whole GiB, rounded down. `0` on a host
    /// whose totals could not be read.
    pub fn total_gb(&self) -> u64 {
        self.total_bytes / (1024 * 1024 * 1024)
    }
}

/// Read host RAM totals from `/proc/meminfo`.
///
/// Returns `None` when `/proc/meminfo` is missing (non-Linux) or does
/// not parse, so callers fall back to conservative fixed sizing rather
/// than to a guess.
pub fn mem_info() -> Option<MemInfo> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// Parse `MemTotal:` / `MemAvailable:` lines (values are in kB).
fn parse_meminfo(text: &str) -> Option<MemInfo> {
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_meminfo_kb(rest);
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }
    Some(MemInfo {
        total_bytes: total? * 1024,
        available_bytes: available? * 1024,
    })
}

fn parse_meminfo_kb(rest: &str) -> Option<u64> {
    rest.trim().strip_suffix("kB")?.trim().parse().ok()
}

/// How many NUMA nodes the kernel reports, or `None` off Linux / when
/// the topology is not exposed.
///
/// Matters for one specific case: a **multi-package APU**. Each package
/// carries its own memory and appears as its own NUMA node, while
/// [`mem_info`] reads `/proc/meminfo`, which is SYSTEM-WIDE. So on such
/// a machine "subtract this GPU's memory from host RAM" is wrong in both
/// directions — every rank would either subtract its own package's share
/// from the whole-system total (collapsing the budget to nothing) or
/// treat the whole-system total as its own (over-committing per package).
///
/// There is no per-node `MemAvailable` to fall back on: the kernel only
/// computes that estimate system-wide, so `/sys/devices/system/node/N/`
/// offers `MemFree` and nothing equivalent. Hence detect-and-refuse
/// rather than detect-and-approximate.
///
/// A count of 1 (the overwhelmingly common case, including every
/// consumer APU) means the system-wide figures ARE the per-node figures
/// and the budget math is sound.
pub fn numa_node_count() -> Option<usize> {
    let dir = std::fs::read_dir("/sys/devices/system/node").ok()?;
    let n = dir
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .and_then(|s| s.strip_prefix("node"))
                .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
        })
        .count();
    (n > 0).then_some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_meminfo_format() {
        let text = "MemTotal:       131781120 kB\n\
                    MemFree:         8123456 kB\n\
                    MemAvailable:   98765432 kB\n\
                    Buffers:          123456 kB\n";
        let m = parse_meminfo(text).unwrap();
        assert_eq!(m.total_bytes, 131_781_120 * 1024);
        assert_eq!(m.available_bytes, 98_765_432 * 1024);
        assert_eq!(m.total_gb(), 125);
    }

    #[test]
    fn missing_memavailable_is_none_not_a_guess() {
        // Pre-3.14 kernels lack MemAvailable. Refusing to answer beats
        // inventing a number the budget logic would trust.
        assert!(parse_meminfo("MemTotal: 100 kB\nMemFree: 50 kB\n").is_none());
        assert!(parse_meminfo("").is_none());
    }

    #[test]
    fn reads_live_host() {
        // On Linux this must parse; elsewhere None is the contract.
        if let Some(m) = mem_info() {
            assert!(m.total_bytes > 0);
            assert!(m.available_bytes <= m.total_bytes);
        }
    }
}
