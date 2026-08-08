//! Variant management: remove installed libtorch variants.

use std::fs;
use std::path::Path;

use super::detect;

/// Remove an installed libtorch variant.
///
/// If the removed variant was active, clears `libtorch/.active`.
pub fn remove_variant(root: &Path, variant: &str) -> Result<(), String> {
    if !detect::is_valid_variant(root, variant) {
        return Err(format!(
            "'{}' is not an installed libtorch variant.\n\
             Expected: libtorch/{}/lib/ to exist.",
            variant, variant
        ));
    }

    let variant_dir = root.join(format!("libtorch/{}", variant));

    // Check if this is the active variant
    let was_active = detect::read_active(root).is_some_and(|info| info.path == variant);

    // Remove the directory
    fs::remove_dir_all(&variant_dir)
        .map_err(|e| format!("cannot remove {}: {}", variant_dir.display(), e))?;

    println!("  Removed: {}", variant);

    // Clear .active if this was the active variant
    if was_active {
        let active_path = root.join("libtorch/.active");
        let _ = fs::write(&active_path, "");
        println!("  Cleared active variant (was pointing to removed variant).");
        println!("  Run 'fdl libtorch activate <variant>' to set a new one.");
    }

    Ok(())
}
