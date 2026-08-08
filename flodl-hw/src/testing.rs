//! Hardware spoofing for tests: [`ENV_TESTING_GPU_JSON`].
//!
//! Detection here is subprocess-and-filesystem based, so it can be
//! replaced wholesale by one env var. That makes a second vendor's
//! detection, libtorch variant selection and per-host prebuild routing
//! testable on a machine that has none of that hardware.
//!
//! Sibling of `FLODL_TESTING_CLUSTER_JSON` (which injects a cluster
//! topology), with one deliberate difference: **no hex encoding**. The
//! cluster envelope is written machine-to-machine by `fdl`, so hex
//! costs nothing there; this one is meant to be typed by a developer,
//! so it takes plain JSON.
//!
//! # Format
//!
//! Either a bare array of devices:
//!
//! ```text
//! FLODL_TESTING_GPU_JSON='[{"vendor":"amd","arch":"gfx1030","vram_mb":16384,"name":"AMD Radeon RX 6800"}]'
//! ```
//!
//! or an envelope, when the point of the test is a *finding* rather
//! than a device. The AMD-card-without-ROCm case has **no** device, and
//! its message is the whole behaviour under test:
//!
//! ```text
//! FLODL_TESTING_GPU_JSON='{"gpus":[],"notes":[{"vendor":"amd","kind":"hardware_unusable","message":"..."}]}'
//! ```
//!
//! The array form is the same shape `fdl probe --json` emits under
//! `gpus`, so a report captured from real hardware replays after a
//! `jq '.gpus'`. That matters: AMD detection is being built against
//! synthetic fixtures, and the first real capture should drop in with
//! as little reshaping as possible.
//!
//! Device fields: `arch` is required (there is no safe default for it);
//! `vendor` defaults to `nvidia`, `index` to the array position, `name`
//! to a generated label, `vram_mb` to `0`. `sm` is accepted as a legacy
//! alias for `arch`. Unknown keys are an error, so a typo fails loudly
//! instead of silently taking a default.
//!
//! # Masks still apply
//!
//! The spoof replaces the *hardware*, not the visibility policy:
//! [`crate::survey`] returns it verbatim, and [`crate::survey_visible`]
//! then applies `CUDA_VISIBLE_DEVICES` on top. Spoofing four devices
//! and masking to one is a valid, and useful, combination.
//!
//! # Failure behaviour
//!
//! A malformed value **panics**. This variable is only ever set
//! deliberately, and a test that silently fell back to real hardware
//! after a typo would report on the wrong machine. Same reasoning as
//! `discover_test_cluster`'s loud parse failure.

use serde_json::Value;

use crate::GpuInfo;
use crate::report::{GpuSurvey, NoteKind, SurveyNote};
use crate::vendor::{GpuArch, GpuVendor};

/// Env var carrying a spoofed GPU survey. See the module docs.
pub const ENV_TESTING_GPU_JSON: &str = "FLODL_TESTING_GPU_JSON";

/// The spoofed survey, when [`ENV_TESTING_GPU_JSON`] is set.
///
/// Returns `None` when the variable is unset or empty, so real
/// detection runs. Panics when it is set but malformed.
pub(crate) fn spoofed_survey() -> Option<GpuSurvey> {
    let raw = std::env::var(ENV_TESTING_GPU_JSON).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    match parse_survey(&raw) {
        Ok(s) => Some(s),
        Err(e) => panic!(
            "{ENV_TESTING_GPU_JSON} is set but could not be parsed: {e}\n\
             Expected a JSON array of devices, e.g.\n  \
             [{{\"vendor\":\"amd\",\"arch\":\"gfx1030\",\"vram_mb\":16384}}]\n\
             or an envelope {{\"gpus\":[...],\"notes\":[...]}}.\n\
             Unset the variable to use real hardware detection."
        ),
    }
}

/// Parse the env value into a survey. Split out from
/// [`spoofed_survey`] so every branch is testable without touching
/// process environment.
fn parse_survey(raw: &str) -> Result<GpuSurvey, String> {
    let root: Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    let empty: Vec<Value> = Vec::new();
    let (devices, notes) = match &root {
        Value::Array(items) => (items, &empty),
        Value::Object(_) => {
            let devices = match field(&root, "gpus") {
                Some(Value::Array(items)) => items,
                Some(other) => {
                    return Err(format!("`gpus` must be an array, found {}", kind(other)));
                }
                None => &empty,
            };
            let notes = match field(&root, "notes") {
                Some(Value::Array(items)) => items,
                Some(other) => {
                    return Err(format!("`notes` must be an array, found {}", kind(other)));
                }
                None => &empty,
            };
            (devices, notes)
        }
        other => {
            return Err(format!(
                "expected an array of devices or a {{\"gpus\":[...]}} envelope, found {}",
                kind(other)
            ));
        }
    };

    let mut out = GpuSurvey::default();
    for (pos, d) in devices.iter().enumerate() {
        out.devices.push(parse_device(d, pos)?);
    }
    for n in notes {
        out.notes.push(parse_note(n)?);
    }
    Ok(out)
}

/// Look up a key, treating an explicit `null` as absent so a captured
/// record with null fields takes the same defaults a hand-written one
/// would.
fn field<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key).filter(|v| !v.is_null())
}

/// Type name for error messages.
fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

const DEVICE_KEYS: &[&str] = &["index", "vendor", "name", "arch", "sm", "vram_mb"];
const NOTE_KEYS: &[&str] = &["vendor", "kind", "message"];

/// Reject unknown keys. A silently-ignored typo (`"vram"` for
/// `"vram_mb"`) would leave the field at its default and make the test
/// assert against a device its author never described.
fn reject_unknown_keys(v: &Value, allowed: &[&str], what: &str) -> Result<(), String> {
    if let Value::Object(map) = v {
        for k in map.keys() {
            if !allowed.contains(&k.as_str()) {
                return Err(format!(
                    "unknown {what} key {k:?} (allowed: {})",
                    allowed.join(", ")
                ));
            }
        }
    }
    Ok(())
}

fn parse_device(v: &Value, pos: usize) -> Result<GpuInfo, String> {
    if !v.is_object() {
        return Err(format!("each device must be an object, found {}", kind(v)));
    }
    reject_unknown_keys(v, DEVICE_KEYS, "device")?;

    let vendor = match field(v, "vendor") {
        None => GpuVendor::Nvidia,
        Some(j) => {
            let s = j.as_str().ok_or_else(|| {
                format!("device {pos}: `vendor` must be a string, found {}", kind(j))
            })?;
            GpuVendor::parse(s).ok_or_else(|| format!("device {pos}: unknown vendor {s:?}"))?
        }
    };

    // `arch` has no safe default: fabricating one would make the device
    // compare as incompatible with every libtorch variant, which reads
    // as a hardware problem rather than a malformed fixture.
    let raw_arch = field(v, "arch")
        .or_else(|| field(v, "sm"))
        .ok_or_else(|| format!("device {pos}: missing required `arch`"))?;
    let token = raw_arch.as_str().ok_or_else(|| {
        format!(
            "device {pos}: `arch` must be a string, found {}",
            kind(raw_arch)
        )
    })?;
    let arch = GpuArch::parse(vendor, token)
        .ok_or_else(|| format!("device {pos}: {token:?} is not a valid {vendor} arch"))?;

    let index = match field(v, "index") {
        None => u8::try_from(pos).map_err(|_| {
            format!("device {pos}: array position exceeds the device-index range (0..255)")
        })?,
        Some(j) => {
            let n = j.as_u64().ok_or_else(|| {
                format!(
                    "device {pos}: `index` must be a non-negative integer, found {}",
                    kind(j)
                )
            })?;
            u8::try_from(n).map_err(|_| {
                format!("device {pos}: `index` {n} exceeds the device-index range (0..255)")
            })?
        }
    };

    let total_memory_mb = match field(v, "vram_mb") {
        None => 0,
        Some(j) => j.as_u64().ok_or_else(|| {
            format!(
                "device {pos}: `vram_mb` must be a non-negative integer, found {}",
                kind(j)
            )
        })?,
    };

    let name = match field(v, "name") {
        None => format!("{vendor} spoofed {arch}"),
        Some(j) => j
            .as_str()
            .ok_or_else(|| format!("device {pos}: `name` must be a string, found {}", kind(j)))?
            .to_string(),
    };

    Ok(GpuInfo {
        index,
        vendor,
        name,
        arch,
        total_memory_mb,
    })
}

fn parse_note(v: &Value) -> Result<SurveyNote, String> {
    if !v.is_object() {
        return Err(format!("each note must be an object, found {}", kind(v)));
    }
    reject_unknown_keys(v, NOTE_KEYS, "note")?;

    let vendor = match field(v, "vendor") {
        None => GpuVendor::Nvidia,
        Some(j) => {
            let s = j
                .as_str()
                .ok_or_else(|| format!("note: `vendor` must be a string, found {}", kind(j)))?;
            GpuVendor::parse(s).ok_or_else(|| format!("note: unknown vendor {s:?}"))?
        }
    };
    let note_kind = match field(v, "kind") {
        None => NoteKind::HardwareUnusable,
        Some(j) => {
            let s = j
                .as_str()
                .ok_or_else(|| format!("note: `kind` must be a string, found {}", kind(j)))?;
            NoteKind::parse(s).ok_or_else(|| {
                format!(
                    "note: unknown kind {s:?} (allowed: {})",
                    NoteKind::ALL_NAMES
                )
            })?
        }
    };
    let message = match field(v, "message") {
        None => return Err("note: missing required `message`".to_string()),
        Some(j) => j
            .as_str()
            .ok_or_else(|| format!("note: `message` must be a string, found {}", kind(j)))?
            .to_string(),
    };
    Ok(SurveyNote {
        vendor,
        kind: note_kind,
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_array_spoofs_devices() {
        let s = parse_survey(
            r#"[{"vendor":"amd","arch":"gfx1030","vram_mb":16384,"name":"AMD Radeon RX 6800"},
                {"vendor":"nvidia","arch":"sm_120","vram_mb":16311,"index":3}]"#,
        )
        .unwrap();
        assert_eq!(s.devices.len(), 2);
        assert_eq!(s.devices[0].vendor, GpuVendor::Amd);
        assert_eq!(s.devices[0].arch, GpuArch::Gfx("gfx1030".into()));
        assert_eq!(s.devices[0].index, 0, "index defaults to array position");
        assert_eq!(s.devices[0].short_name(), "Radeon RX 6800");
        assert_eq!(s.devices[1].index, 3, "explicit index wins");
        assert_eq!(
            s.devices[1].arch,
            GpuArch::Sm {
                major: 12,
                minor: 0
            }
        );
        assert!(s.notes.is_empty());
    }

    #[test]
    fn replays_a_captured_probe_gpus_array() {
        // This is the shape `fdl probe --json` emits under `gpus`,
        // legacy `sm` key and all. A capture from real AMD hardware
        // must drop in after `jq '.gpus'`, since that is how the
        // synthetic P2 fixtures eventually get replaced.
        let s = parse_survey(
            r#"[{"index":0,"name":"NVIDIA GeForce RTX 5060 Ti","vendor":"nvidia",
                 "arch":"sm_120","sm":"sm_120","vram_mb":16311}]"#,
        )
        .unwrap();
        assert_eq!(s.devices.len(), 1);
        assert_eq!(
            s.devices[0].arch,
            GpuArch::Sm {
                major: 12,
                minor: 0
            }
        );
        assert_eq!(s.devices[0].total_memory_mb, 16311);
    }

    #[test]
    fn sm_alone_works_for_a_legacy_capture() {
        let s = parse_survey(r#"[{"sm":"sm_86","vram_mb":24564}]"#).unwrap();
        assert_eq!(s.devices[0].arch, GpuArch::Sm { major: 8, minor: 6 });
        assert_eq!(s.devices[0].vendor, GpuVendor::Nvidia);
    }

    #[test]
    fn envelope_carries_notes_with_no_devices() {
        // The case the whole survey type exists for: hardware present,
        // stack not installed. There is no device to describe, so the
        // note IS the behaviour under test.
        let s = parse_survey(
            r#"{"gpus":[],"notes":[{"vendor":"amd","kind":"hardware_unusable",
                 "message":"an AMD GPU is present but ROCm is not installed"}]}"#,
        )
        .unwrap();
        assert!(s.devices.is_empty());
        assert_eq!(s.notes.len(), 1);
        assert_eq!(s.notes[0].vendor, GpuVendor::Amd);
        assert_eq!(s.notes[0].kind, NoteKind::HardwareUnusable);
        let err = s.require_devices().unwrap_err();
        assert!(err.contains("ROCm is not installed"), "got: {err}");
    }

    #[test]
    fn envelope_fields_are_optional() {
        assert!(parse_survey("{}").unwrap().devices.is_empty());
        assert!(parse_survey(r#"{"gpus":[]}"#).unwrap().notes.is_empty());
        assert!(parse_survey("[]").unwrap().devices.is_empty());
    }

    #[test]
    fn explicit_null_reads_as_absent() {
        // A captured record with null fields must take the same
        // defaults a hand-written one would, not error.
        let s = parse_survey(r#"[{"arch":"sm_86","name":null,"index":null}]"#).unwrap();
        assert_eq!(s.devices[0].index, 0);
        assert_eq!(s.devices[0].name, "NVIDIA spoofed sm_86");
    }

    #[test]
    fn a_name_is_generated_when_omitted() {
        let s = parse_survey(r#"[{"vendor":"amd","arch":"gfx942"}]"#).unwrap();
        assert_eq!(s.devices[0].name, "AMD spoofed gfx942");
    }

    #[test]
    fn missing_arch_is_an_error_not_a_default() {
        // A fabricated arch compares as incompatible with every
        // libtorch variant, which reads as a hardware problem the
        // fixture author does not have.
        let e = parse_survey(r#"[{"vendor":"amd","vram_mb":8192}]"#).unwrap_err();
        assert!(e.contains("missing required `arch`"), "got: {e}");
    }

    #[test]
    fn arch_must_match_its_vendors_shape() {
        let e = parse_survey(r#"[{"vendor":"amd","arch":"sm_120"}]"#).unwrap_err();
        assert!(e.contains("not a valid AMD arch"), "got: {e}");
        let e = parse_survey(r#"[{"vendor":"nvidia","arch":"gfx1030"}]"#).unwrap_err();
        assert!(e.contains("not a valid NVIDIA arch"), "got: {e}");
    }

    #[test]
    fn a_typo_in_a_key_fails_loudly() {
        // `vram` silently defaulting to 0 would make the test assert
        // against a device its author never described.
        let e = parse_survey(r#"[{"arch":"sm_86","vram":8192}]"#).unwrap_err();
        assert!(e.contains("unknown device key \"vram\""), "got: {e}");
    }

    #[test]
    fn rejects_wrong_types_and_out_of_range_values() {
        for (input, want) in [
            (
                r#"[{"arch":"sm_86","index":"0"}]"#,
                "must be a non-negative integer",
            ),
            (
                r#"[{"arch":"sm_86","index":300}]"#,
                "exceeds the device-index range",
            ),
            (
                r#"[{"arch":"sm_86","index":-1}]"#,
                "must be a non-negative integer",
            ),
            (
                r#"[{"arch":"sm_86","vram_mb":-5}]"#,
                "must be a non-negative integer",
            ),
            (
                r#"[{"arch":"sm_86","vram_mb":1.5}]"#,
                "must be a non-negative integer",
            ),
            (r#"[{"arch":"sm_86","name":7}]"#, "`name` must be a string"),
            (r#"[{"arch":7}]"#, "`arch` must be a string"),
            (r#"[{"vendor":"intel","arch":"sm_86"}]"#, "unknown vendor"),
            (r#"["sm_86"]"#, "must be an object"),
            (r#"{"gpus":7}"#, "`gpus` must be an array"),
            (r#"{"notes":7}"#, "`notes` must be an array"),
            (
                r#"{"notes":[{"vendor":"amd"}]}"#,
                "missing required `message`",
            ),
            (
                r#"{"notes":[{"kind":"nope","message":"m"}]}"#,
                "unknown kind",
            ),
            ("7", "expected an array of devices"),
            ("\"hi\"", "expected an array of devices"),
        ] {
            let e = parse_survey(input).unwrap_err();
            assert!(
                e.contains(want),
                "input {input}\n  got: {e}\n  want: {want}"
            );
        }
    }

    #[test]
    fn malformed_json_is_rejected_by_serde() {
        // serde_json owns the grammar (including a recursion limit, so
        // deeply-nested input errors instead of overflowing the stack).
        // These assert the wiring surfaces its error, not that we
        // reimplemented a parser.
        for bad in ["", "[", "[1,", "{\"a\" 1}", "[1] trailing", "{\"a\":1,}"] {
            assert!(parse_survey(bad).is_err(), "should reject {bad:?}");
        }
        let deep = format!("{}{}", "[".repeat(300), "]".repeat(300));
        assert!(
            parse_survey(&deep).is_err(),
            "recursion limit must reject, not overflow"
        );
    }

    #[test]
    fn note_kind_round_trips_through_its_name() {
        for k in [
            NoteKind::HardwareUnusable,
            NoteKind::ToolFailed,
            NoteKind::Unparsable,
            NoteKind::MaskApplied,
        ] {
            assert_eq!(NoteKind::parse(k.as_str()), Some(k));
            assert!(NoteKind::ALL_NAMES.contains(k.as_str()), "{k:?} listed");
        }
    }
}
