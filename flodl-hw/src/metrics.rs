//! Live per-device metrics: utilisation, temperature and power.
//!
//! Detection ([`crate::survey`]) answers "what is installed". This
//! module answers "what is it doing right now", which is a different
//! question with a different cost profile: it is read on a timer for
//! the life of a training run, so path resolution happens **once** and
//! each tick is a bare file read.
//!
//! # AMD only, and that is not an omission
//!
//! NVIDIA exposes no equivalent through sysfs -- an `nvidia`-driven DRM
//! card carries no `gpu_busy_percent` -- so NVIDIA live metrics come
//! from NVML, which lives behind the FFI shim rather than here (this
//! crate loads no GPU runtime). There is no unification to be had: the
//! two vendors are two backends either way, exactly as they already are
//! for detection.
//!
//! # Why the render minor, not `/sys/class/drm/card*`
//!
//! A KFD node publishes `drm_render_minor`, so
//! `/sys/class/drm/renderD<minor>/device` reaches the very same
//! directory as `card<n>/device` (both are symlinks to the PCI device)
//! with no globbing and no matching step. That matters twice: it is
//! exact where a `card*` scan would have to disambiguate several AMD
//! cards by PCI address, and it is available where a card node might
//! not be, since a headless compute part is a render node first.
//! Position in [`amd_metrics_probes`] is therefore device index by
//! construction: it walks the same filtered, sorted node list detection
//! walks.
//!
//! # In a container, the KFD tree needs `/dev/kfd` passed through
//!
//! Every DRM, hwmon and PCI attribute this module reads is world
//! readable and reads fine inside a stock container. The KFD topology
//! does **not**: it returns `EPERM` unless the container was given
//! `--device=/dev/kfd`, and no `--privileged` or AppArmor change
//! substitutes for it. That is not a limitation worth working around,
//! because the same device node is what ROCm itself requires: a
//! container that cannot read the topology could not have trained on
//! the card anyway, so resolving no probes there is the correct answer
//! rather than a missing fallback. It is worth knowing when a
//! containerised run reports no AMD metrics, since the cause is
//! upstream of this module.
//!
//! # Attribute names vary by ASIC, so nothing here is hardcoded
//!
//! Verified on a live gfx1036: that part exposes `power1_input` and
//! **not** `power1_average`, no `power1_cap`, and only `temp1_input`
//! (labelled `edge`) where a discrete part typically adds junction and
//! memory sensors. So the probe resolves whatever the driver published
//! and records it; an attribute that is not there reads `None` rather
//! than zero, which the whole consuming chain already handles.

use std::path::{Path, PathBuf};

/// A device's live metrics.
///
/// Every field is optional and absent means "not published by this
/// driver on this part", never zero. Units are normalised here (the
/// kernel reports millidegrees and microwatts) so callers never carry
/// a scale factor.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuMetrics {
    /// Compute utilisation, 0-100.
    pub util_percent: Option<u8>,
    /// Hottest published sensor, in degrees Celsius.
    ///
    /// The **maximum** across every `temp*_input` the part exposes, not
    /// the edge sensor specifically: parts differ in which sensors they
    /// publish, and the question this answers ("how hot is the hottest
    /// point on this card") is the one that survives that difference.
    pub temp_c: Option<f32>,
    /// Instantaneous board power draw, in watts.
    pub power_w: Option<f32>,
}

impl GpuMetrics {
    /// Whether the driver published anything at all.
    pub fn is_empty(&self) -> bool {
        self.util_percent.is_none() && self.temp_c.is_none() && self.power_w.is_none()
    }
}

/// Resolved sysfs paths for one AMD GPU's live metrics.
///
/// Construction does every directory read; [`AmdMetricsProbe::read`]
/// only opens files that were found to exist. Cheap enough to poll at
/// a few hertz and safe to hold for a whole run: the paths are stable
/// for as long as the device is bound.
#[derive(Debug, Clone, Default)]
pub struct AmdMetricsProbe {
    busy: Option<PathBuf>,
    /// Every `temp*_input` found, sorted so the set is deterministic.
    temps: Vec<PathBuf>,
    power: Option<PathBuf>,
}

impl AmdMetricsProbe {
    /// Whether this probe resolved any attribute. A probe that resolved
    /// nothing still reads successfully; it just always returns an empty
    /// [`GpuMetrics`].
    pub fn is_empty(&self) -> bool {
        self.busy.is_none() && self.temps.is_empty() && self.power.is_none()
    }

    /// Sample every attribute that resolved.
    ///
    /// A read that fails is `None` for that field only. Failure is
    /// normal rather than exceptional: several amdgpu attributes return
    /// an error while the part is power-gated, and a suspended device
    /// must not take the whole sample down.
    pub fn read(&self) -> GpuMetrics {
        GpuMetrics {
            // Clamped rather than trusted: the field is a percentage by
            // ABI, and a consumer computing an idle fraction from it
            // should not have to defend against a driver bug.
            util_percent: self
                .busy
                .as_deref()
                .and_then(read_u64)
                .map(|v| v.min(100) as u8),
            temp_c: self
                .temps
                .iter()
                .filter_map(|p| read_u64(p))
                .max()
                .map(|milli| milli as f32 / 1000.0),
            power_w: self
                .power
                .as_deref()
                .and_then(read_u64)
                .map(|micro| micro as f32 / 1_000_000.0),
        }
    }
}

/// Resolve a live-metrics probe for every AMD GPU on this host, in
/// device-index order.
///
/// Index N of the result describes the same device as index N of the
/// AMD half of [`crate::detect_gpus_physical`]. Visibility masks are
/// deliberately **not** applied: this mirrors NVML, which also reports
/// by physical index, so a caller holding a device's physical index can
/// look up either backend the same way.
///
/// Empty on a host with no AMD GPU, which is not an error.
pub fn amd_metrics_probes() -> Vec<AmdMetricsProbe> {
    amd_metrics_probes_at(Path::new("/sys"))
}

/// [`amd_metrics_probes`] with its sysfs root injected, so path
/// resolution is testable against a synthetic tree.
pub(crate) fn amd_metrics_probes_at(sys: &Path) -> Vec<AmdMetricsProbe> {
    crate::amd::gpu_node_dirs(&sys.join("class/kfd/kfd/topology/nodes"))
        .iter()
        .map(|node| resolve(sys, node))
        .collect()
}

/// Resolve one node's attribute paths.
fn resolve(sys: &Path, node_dir: &Path) -> AmdMetricsProbe {
    let mut probe = AmdMetricsProbe::default();

    let Ok(props) = std::fs::read_to_string(node_dir.join("properties")) else {
        return probe;
    };
    // 0 means the kernel published no render node for this device (it
    // is what the CPU node reports), so there is nothing to resolve.
    let minor = match crate::amd::prop(&props, "drm_render_minor") {
        Some(m) if m > 0 => m,
        _ => return probe,
    };

    let device = sys.join(format!("class/drm/renderD{minor}/device"));
    let busy = device.join("gpu_busy_percent");
    if busy.exists() {
        probe.busy = Some(busy);
    }

    let Some(hwmon) = amdgpu_hwmon(&device.join("hwmon")) else {
        return probe;
    };
    let Ok(entries) = std::fs::read_dir(&hwmon) else {
        return probe;
    };
    // One pass over the directory: the names present are the ASIC's
    // answer to what it can measure, so this reads the set rather than
    // testing for a fixed list of candidates.
    let mut power_input = None;
    for entry in entries.flatten() {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        match name.as_str() {
            // The smoothed reading where a part offers it; `_input` is
            // instantaneous and is the fallback, not the equal.
            "power1_average" => probe.power = Some(entry.path()),
            "power1_input" => power_input = Some(entry.path()),
            _ if name.starts_with("temp") && name.ends_with("_input") => {
                probe.temps.push(entry.path())
            }
            _ => {}
        }
    }
    probe.power = probe.power.or(power_input);
    // `read_dir` order is filesystem order; sort so a probe built twice
    // on the same box is identical, and so the max is over a stable set.
    probe.temps.sort();
    probe
}

/// The `hwmon<N>` directory under a device, when it belongs to amdgpu.
///
/// The index is not predictable (a live single-AMD-GPU host presented
/// `hwmon8`), so it is discovered rather than assumed. There is exactly
/// one entry per device, and the `name` check keeps this honest if that
/// ever stops being true.
fn amdgpu_hwmon(hwmon_dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(hwmon_dir).ok()?.flatten().find_map(|e| {
        let path = e.path();
        let name = std::fs::read_to_string(path.join("name")).ok()?;
        (name.trim() == "amdgpu").then_some(path)
    })
}

/// Read a sysfs file holding a single unsigned integer.
fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// Scratch dir + RAII cleanup; flodl-hw takes no dev-dependencies.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("flodl-hw-metrics-{nanos}-{seq}"));
            fs::create_dir_all(&dir).expect("scratch");
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A KFD GPU node carrying `drm_render_minor`.
    fn kfd_node(sys: &Path, n: u64, render_minor: u64) {
        let dir = sys.join("class/kfd/kfd/topology/nodes").join(n.to_string());
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("properties"),
            format!(
                "cpu_cores_count 0\nsimd_count 4\ngfx_target_version 100306\n\
                 vendor_id 4098\ndevice_id 5056\nlocation_id 3584\n\
                 drm_render_minor {render_minor}\n"
            ),
        )
        .unwrap();
        fs::write(dir.join("gpu_id"), "6720\n").unwrap();
    }

    fn cpu_node(sys: &Path) {
        let dir = sys.join("class/kfd/kfd/topology/nodes/0");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("properties"),
            "cpu_cores_count 24\nsimd_count 0\ngfx_target_version 0\n\
             vendor_id 0\ndevice_id 0\nlocation_id 0\ndrm_render_minor 0\n",
        )
        .unwrap();
    }

    /// A render node's device dir. `attrs` are written into its hwmon.
    fn render_device(sys: &Path, minor: u64, busy: Option<&str>, attrs: &[(&str, &str)]) {
        let device = sys.join(format!("class/drm/renderD{minor}/device"));
        fs::create_dir_all(&device).unwrap();
        if let Some(v) = busy {
            fs::write(device.join("gpu_busy_percent"), v).unwrap();
        }
        // A non-zero index on purpose: nothing may assume hwmon0.
        let hwmon = device.join("hwmon/hwmon8");
        fs::create_dir_all(&hwmon).unwrap();
        fs::write(hwmon.join("name"), "amdgpu\n").unwrap();
        for (k, v) in attrs {
            fs::write(hwmon.join(k), v).unwrap();
        }
    }

    /// The shape verified on a live gfx1036: `power1_input` and no
    /// `power1_average`, one temperature sensor, hwmon index 8.
    #[test]
    fn resolves_the_live_igpu_shape() {
        let s = Scratch::new();
        cpu_node(s.path());
        kfd_node(s.path(), 1, 129);
        render_device(
            s.path(),
            129,
            Some("37\n"),
            &[
                ("temp1_input", "44000\n"),
                ("temp1_label", "edge\n"),
                ("power1_input", "11000\n"),
                ("power1_label", "PPT\n"),
                ("freq1_input", "600000000\n"),
            ],
        );

        let probes = amd_metrics_probes_at(s.path());
        assert_eq!(probes.len(), 1, "the CPU node is not a device");
        let m = probes[0].read();
        assert_eq!(m.util_percent, Some(37));
        assert_eq!(m.temp_c, Some(44.0));
        assert_eq!(m.power_w, Some(0.011));
    }

    /// `power1_average` is preferred where a part offers both, since
    /// `_input` is instantaneous and noisier at a sampler's cadence.
    #[test]
    fn average_power_wins_over_instantaneous_when_both_exist() {
        let s = Scratch::new();
        kfd_node(s.path(), 1, 128);
        render_device(
            s.path(),
            128,
            Some("50\n"),
            &[
                ("power1_average", "15000000\n"),
                ("power1_input", "16000000\n"),
            ],
        );
        assert_eq!(
            amd_metrics_probes_at(s.path())[0].read().power_w,
            Some(15.0)
        );
    }

    /// The hottest sensor wins, because parts differ in which they
    /// publish and a mean over an inconsistent set means nothing.
    #[test]
    fn temperature_is_the_max_across_every_sensor() {
        let s = Scratch::new();
        kfd_node(s.path(), 1, 128);
        render_device(
            s.path(),
            128,
            None,
            &[
                ("temp1_input", "44000\n"), // edge
                ("temp2_input", "61000\n"), // junction, the hot one
                ("temp3_input", "52000\n"), // memory
                ("temp2_label", "junction\n"),
            ],
        );
        let m = amd_metrics_probes_at(s.path())[0].read();
        assert_eq!(m.temp_c, Some(61.0));
        assert_eq!(m.util_percent, None, "absent attribute is None, not 0");
    }

    /// A part that publishes nothing still probes and reads cleanly.
    #[test]
    fn a_device_with_no_published_attributes_reads_empty_not_zero() {
        let s = Scratch::new();
        kfd_node(s.path(), 1, 128);
        let device = s.path().join("class/drm/renderD128/device");
        fs::create_dir_all(&device).unwrap();

        let probes = amd_metrics_probes_at(s.path());
        assert!(probes[0].is_empty());
        assert!(probes[0].read().is_empty());
    }

    /// Probe order must match detection order, or one device's
    /// temperature is reported against another's name.
    #[test]
    fn probe_index_follows_device_index_not_node_number() {
        let s = Scratch::new();
        cpu_node(s.path());
        // Node numbers 2 and 10: device 0 is node 2, and a lexical sort
        // would put node 10 first.
        kfd_node(s.path(), 2, 128);
        kfd_node(s.path(), 10, 129);
        render_device(s.path(), 128, Some("11\n"), &[]);
        render_device(s.path(), 129, Some("22\n"), &[]);

        let probes = amd_metrics_probes_at(s.path());
        assert_eq!(probes.len(), 2);
        assert_eq!(probes[0].read().util_percent, Some(11));
        assert_eq!(probes[1].read().util_percent, Some(22));
    }

    /// A foreign hwmon under the device must not be read as the GPU's.
    #[test]
    fn a_non_amdgpu_hwmon_is_not_adopted() {
        let s = Scratch::new();
        kfd_node(s.path(), 1, 128);
        let device = s.path().join("class/drm/renderD128/device");
        let hwmon = device.join("hwmon/hwmon3");
        fs::create_dir_all(&hwmon).unwrap();
        fs::write(hwmon.join("name"), "k10temp\n").unwrap();
        fs::write(hwmon.join("temp1_input"), "95000\n").unwrap();

        assert_eq!(amd_metrics_probes_at(s.path())[0].read().temp_c, None);
    }

    /// A device whose kernel published no render node resolves nothing
    /// rather than pointing at `renderD0`.
    #[test]
    fn a_missing_render_minor_resolves_nothing() {
        let s = Scratch::new();
        kfd_node(s.path(), 1, 0);
        render_device(s.path(), 0, Some("99\n"), &[("temp1_input", "44000\n")]);
        assert!(amd_metrics_probes_at(s.path())[0].is_empty());
    }

    /// A driver reporting out of range must not produce an idle
    /// fraction above 100%.
    #[test]
    fn utilisation_is_clamped_to_a_percentage() {
        let s = Scratch::new();
        kfd_node(s.path(), 1, 128);
        render_device(s.path(), 128, Some("250\n"), &[]);
        assert_eq!(
            amd_metrics_probes_at(s.path())[0].read().util_percent,
            Some(100)
        );
    }

    /// A power-gated part errors on a read that resolved at probe time;
    /// that field goes None and the others still report.
    #[test]
    fn an_unreadable_attribute_does_not_take_the_sample_down() {
        let s = Scratch::new();
        kfd_node(s.path(), 1, 128);
        render_device(
            s.path(),
            128,
            Some("5\n"),
            &[("temp1_input", "44000\n"), ("power1_input", "\n")],
        );
        let m = amd_metrics_probes_at(s.path())[0].read();
        assert_eq!(m.power_w, None, "unparseable is None");
        assert_eq!(m.util_percent, Some(5));
        assert_eq!(m.temp_c, Some(44.0));
    }

    /// No AMD hardware is not an error.
    #[test]
    fn a_host_with_no_amd_gpu_yields_no_probes() {
        let s = Scratch::new();
        cpu_node(s.path());
        assert!(amd_metrics_probes_at(s.path()).is_empty());
    }

    /// Report what this box actually publishes.
    ///
    /// Every test above runs against a synthetic tree, which proves the
    /// path composition and nothing about a driver. `#[ignore]` because
    /// it asserts nothing and needs AMD hardware; run it explicitly on
    /// new silicon, where the attribute set is the open question:
    ///
    /// ```text
    /// cargo test -p flodl-hw reports_this_hosts_amd_metrics -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "hardware probe: prints what this host publishes, asserts nothing"]
    fn reports_this_hosts_amd_metrics() {
        let probes = amd_metrics_probes();
        println!("AMD devices with a metrics probe: {}", probes.len());
        for (i, p) in probes.iter().enumerate() {
            println!("  device {i}: {:?}", p.read());
            println!("    resolved: {p:?}");
        }
    }
}
