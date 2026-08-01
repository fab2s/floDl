use std::env;
use std::path::PathBuf;

fn main() {
    // docs.rs builds without libtorch — skip C++ compilation entirely.
    // cargo doc does not link, so unresolved extern symbols are fine.
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    // --- vendor selection -------------------------------------------
    //
    // `gpu` is the vendor-neutral gate every GPU code path uses; `cuda`
    // and `rocm` are the selectors that decide what gets linked. Cargo
    // features are additive and cannot be made exclusive, so the
    // combinations that make no sense are rejected here.
    //
    // Both checks sit AFTER the DOCS_RS early return above, so an
    // `--all-features` documentation build (which necessarily enables
    // both vendors) is unaffected.
    let want_cuda = cfg!(feature = "cuda");
    let want_rocm = cfg!(feature = "rocm");
    if want_cuda && want_rocm {
        eprintln!(
            "\nflodl-sys: features `cuda` and `rocm` are mutually exclusive.\n\
             They select which libtorch backend to link against, and a build\n\
             can only link one. Enable exactly one.\n"
        );
        std::process::exit(1);
    }
    if cfg!(feature = "gpu") && !want_cuda && !want_rocm {
        eprintln!(
            "\nflodl-sys: feature `gpu` is enabled with no vendor selected.\n\
             `gpu` marks the vendor-neutral GPU code paths; it does not say\n\
             what to link. Enable `cuda` or `rocm` instead -- both imply it.\n"
        );
        std::process::exit(1);
    }
    let libtorch = env::var("LIBTORCH_PATH")
        .unwrap_or_else(|_| "/usr/local/libtorch".to_string());
    let libtorch = PathBuf::from(&libtorch);

    // Preflight: confirm libtorch is actually present before cc::Build
    // launches a multi-minute compile that would otherwise fail with a
    // cryptic `fatal error: torch/torch.h: No such file or directory`
    // deep in the C++ output. Pointing users at `fdl setup` is the
    // canonical fix; the manual override is documented for users who
    // are bypassing fdl on purpose.
    // Match the same header file cc::Build's include path resolves
    // (`include/torch/csrc/api/include`); presence here is the
    // canonical "libtorch is installed" sentinel for both the
    // pre-built and source-built variants.
    let torch_header = libtorch
        .join("include/torch/csrc/api/include/torch/torch.h");
    if !torch_header.exists() {
        eprintln!(
            "\nflodl-sys: libtorch not found at `{}`\n\
             (expected `{}` to exist).\n\n\
             Recommended fix: install `flodl-cli` and run `fdl setup` from\n\
             your project root. It auto-detects your hardware, downloads or\n\
             builds the matching libtorch variant, and points LIBTORCH_PATH\n\
             at it for you.\n\n\
             Manual override: set LIBTORCH_PATH=/path/to/libtorch where the\n\
             directory contains both `include/torch/csrc/api/include/torch/torch.h`\n\
             and `lib/libtorch.so` (or the platform equivalent).\n",
            libtorch.display(),
            torch_header.display(),
        );
        std::process::exit(1);
    }

    // Unity build: shim.cpp #includes the topic-focused ops_*.cpp files so the
    // C++ compiler parses torch/torch.h exactly once. Splitting into separate
    // TUs would multiply torch.h parse cost (~17s/TU) since cc::Build rebuilds
    // every TU on any change.
    //
    // Files listed for cargo:rerun-if-changed below; only shim.cpp is compiled.
    let shim_includes = [
        "shim.h",
        "helpers.h",
        "ops_tensor.cpp",
        "ops_nn.cpp",
        "ops_math_ext.cpp",
        "ops_training.cpp",
        "ops_cuda.cpp",
    ];

    // --- ROCm: proceed only against a libtorch that actually is one ---
    //
    // The refusal is conditional rather than absolute. Compiling and
    // linking the shim needs headers and .so files, NOT a GPU -- only
    // *running* needs silicon -- so a ROCm container with a ROCm
    // libtorch mounted can legitimately build this, and that is how the
    // remaining unknown gets settled. What cannot work is `--features
    // rocm` against a CUDA (or absent) libtorch, and that is the case
    // worth catching early with a message instead of a missing-header
    // error deep in the C++ compile.
    if want_rocm && !libtorch.join("lib/libtorch_hip.so").exists() {
        eprintln!(
            "\nflodl-sys: `--features rocm` needs a ROCm libtorch, but `{}`\n\
             has no `lib/libtorch_hip.so` (so it is a CUDA or CPU build).\n\n\
             Point LIBTORCH_PATH at a ROCm variant -- `fdl libtorch download\n\
             --rocm 6.3` fetches one -- or build with `--features cuda`.\n",
            libtorch.display(),
        );
        std::process::exit(1);
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("shim.cpp")
        .include(".")
        .include(libtorch.join("include"))
        .include(libtorch.join("include/torch/csrc/api/include"))
        .warnings(false);

    // The define gates the shim's GPU code, which is vendor-neutral:
    // ROCm libtorch keeps `kCUDA` and the `c10::cuda` namespaces, so the
    // same blocks serve both backends. Named for what it means.
    if cfg!(feature = "gpu") {
        build.define("FLODL_BUILD_GPU", "1");
    }
    if want_cuda {
        // CUDA toolkit headers (the one genuinely vendor-specific part
        // of the compile step).
        let cuda_home = env::var("CUDA_HOME")
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        build.include(format!("{}/include", cuda_home));
    }
    if want_rocm {
        // ROCm supplies what libtorch-rocm does NOT bundle: the HIP
        // runtime headers (`hip/hip_runtime.h`) and RCCL's `rccl/rccl.h`.
        //
        // libtorch-rocm DOES ship the `c10/cuda/*` and `ATen/cuda/*`
        // header trees, but they are dead weight -- the unbuilt CUDA
        // headers, missing their generated `cuda_cmake_macros.h`, with no
        // `libc10_cuda.so` to link against. `gpu_compat.h` maps onto the
        // `c10/hip/*` + `ATen/hip/*` trees instead. There is likewise no
        // `nccl.h` anywhere in ROCm: RCCL exports the nccl symbol names
        // but ships them as `rccl/rccl.h` only.
        let rocm_path =
            env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        build.include(format!("{rocm_path}/include"));
        // `__HIP_PLATFORM_AMD__` is HIP's own "compiling for AMD" macro,
        // which `gpu_compat.h` keys the whole vendor mapping on.
        build.define("__HIP_PLATFORM_AMD__", "1");
        // `USE_ROCM` is torch's own switch inside the hipified headers.
        // Without it they take their `#else` branch and reach for CUDA
        // headers that a ROCm install does not have -- e.g.
        // `ATen/hip/Exceptions.h` includes `<cusolver_common.h>` unless
        // USE_ROCM is set, and it is reached from `ATen/hip/HIPEvent.h`.
        build.define("USE_ROCM", "1");
    }

    build.compile("flodl_shim");

    // Link libtorch shared libraries
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

    if want_rocm {
        // Link set read off libtorch 2.7.0+rocm6.3's own file list --
        // all four ship inside libtorch/lib, RCCL included.
        println!("cargo:rustc-link-lib=dylib=torch_hip");
        println!("cargo:rustc-link-lib=dylib=c10_hip");
        println!("cargo:rustc-link-lib=dylib=amdhip64");

        let rocm_path =
            env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        println!("cargo:rustc-link-search=native={rocm_path}/lib");

        // dlopen, for the force-load and the allocator probes.
        println!("cargo:rustc-link-lib=dylib=dl");
        // RCCL is NCCL's API-compatible counterpart and exports the same
        // symbol names, so the shim's collective code is unchanged.
        println!("cargo:rustc-link-lib=dylib=rccl");
    }

    if want_cuda {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");

        let cuda_home = env::var("CUDA_HOME")
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
        println!("cargo:rustc-link-lib=dylib=cudart");

        // dlopen for NVML GPU utilization queries
        println!("cargo:rustc-link-lib=dylib=dl");

        // NCCL for multi-GPU collective operations
        println!("cargo:rustc-link-lib=dylib=nccl");
    }

    // Rerun if sources change (shim.cpp + every #included unit + headers).
    println!("cargo:rerun-if-changed=shim.cpp");
    for src in &shim_includes {
        println!("cargo:rerun-if-changed={}", src);
    }
    println!("cargo:rerun-if-env-changed=LIBTORCH_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}
