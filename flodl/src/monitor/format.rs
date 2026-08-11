//! Human-readable formatting for durations, byte counts, and percentages.

/// Format a duration in seconds to a human-readable string.
///
/// Adapts units automatically:
/// - `< 1s` → `"420ms"`
/// - `< 60s` → `"12s"`, `"1.2s"`
/// - `< 1h` → `"4m 32s"`
/// - `≥ 1h` → `"3h 12m"`
pub fn format_eta(secs: f64) -> String {
    if secs < 0.0 {
        return "—".to_string();
    }
    if secs < 1.0 {
        return format!("{}ms", (secs * 1000.0) as u64);
    }
    if secs < 60.0 {
        let whole = secs as u64;
        let frac = ((secs - whole as f64) * 10.0) as u64;
        if frac > 0 {
            return format!("{}.{}s", whole, frac);
        }
        return format!("{}s", whole);
    }
    let total_secs = secs as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs_rem = total_secs % 60;
    if hours > 0 {
        format!("{}h {:02}m", hours, mins)
    } else {
        format!("{}m {:02}s", mins, secs_rem)
    }
}

/// Format a byte count to a human-readable string (e.g., `"2.1 GB"`).
pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;

    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.0} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

/// Format unix seconds as a compact UTC stamp: `YYYYMMDD-HHMMSS`.
///
/// Filesystem-safe (no `:`), lexicographically sortable, and readable in a
/// directory listing — the shape run-scoped telemetry dirs are named with.
/// Civil-calendar math inline so the crate stays chrono-free.
pub fn format_utc_stamp(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // days-since-epoch → (y, m, d), Gregorian (Howard Hinnant's civil_from_days).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}{:02}",
        rem / 3_600,
        (rem % 3_600) / 60,
        rem % 60,
    )
}

/// Adaptive float formatting for metric display.
pub fn format_metric(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".to_string()
    } else if abs < 0.001 {
        format!("{:.2e}", v)
    } else if abs < 100.0 {
        format!("{:.4}", v)
    } else {
        format!("{:.2}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_utc_stamp() {
        assert_eq!(format_utc_stamp(0), "19700101-000000");
        // 2000-03-01 00:00:00 UTC — the day after a century leap day.
        assert_eq!(format_utc_stamp(951_868_800), "20000301-000000");
        // Mid-day with all time components non-zero.
        assert_eq!(format_utc_stamp(951_868_800 + 3_723), "20000301-010203");
    }

    #[test]
    fn test_format_eta() {
        assert_eq!(format_eta(0.042), "42ms");
        assert_eq!(format_eta(1.0), "1s");
        assert_eq!(format_eta(1.5), "1.5s");
        assert_eq!(format_eta(90.0), "1m 30s");
        assert_eq!(format_eta(3661.0), "1h 01m");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024 * 500), "500 KB");
        assert_eq!(format_bytes(1024 * 1024 * 100), "100 MB");
        assert_eq!(format_bytes(2_254_857_830), "2.1 GB");
    }

    #[test]
    fn test_format_metric() {
        assert_eq!(format_metric(0.0), "0");
        assert_eq!(format_metric(0.0001), "1.00e-4");
        assert_eq!(format_metric(1.2345), "1.2345");
        assert_eq!(format_metric(12345.67), "12345.67");
    }
}
