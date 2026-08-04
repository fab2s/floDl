use std::env;
use std::path::{Path, PathBuf};

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
             --rocm 7.0` fetches one -- or build with `--features cuda`.\n",
            libtorch.display(),
        );
        std::process::exit(1);
    }

    // Where each vendor's toolkit lives. Read once: both the guard below
    // and the include setup further down need them. ROCm resolution
    // mirrors flodl-hw's (`$ROCM_PATH` / `$HIP_PATH` / `$HSA_PATH`, then
    // the convention) — build.rs cannot depend on that crate, so the
    // order is kept in sync by hand.
    let rocm_path = ["ROCM_PATH", "HIP_PATH", "HSA_PATH"]
        .iter()
        .filter_map(|k| env::var(k).ok())
        .find(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "/opt/rocm".to_string());
    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Vendor toolkit headers, as (header relative to an include dir,
    // package that owns it). Covers the whole include chain rather than
    // the headers the shim names directly: torch's vendor trees pull in
    // more, so checking only the direct includes passes while the
    // compile still fails.
    //
    // Regenerate after a libtorch bump with `c++ -M` over shim.cpp using
    // the same -I/-D flags set below. Use -M and not -MM: -MM omits
    // system headers, which drops nccl.h (it lives in /usr/include, not
    // under $CUDA_HOME).
    //
    // Kept in sync by hand with flodl-cli's util/requirements.rs;
    // build.rs cannot depend on that crate.
    const ROCM_HEADERS: &[(&str, &str)] = &[
        ("hip/hip_runtime.h", "hip-dev"),
        ("rccl/rccl.h", "rccl-dev"),
        ("hipblas/hipblas.h", "hipblas-dev"),
        ("hipblas-common/hipblas-common.h", "hipblas-common-dev"),
        ("hipblaslt/hipblaslt.h", "hipblaslt-dev"),
        ("hipsolver/hipsolver.h", "hipsolver-dev"),
        ("hipsparse/hipsparse.h", "hipsparse-dev"),
];
    const CUDA_HEADERS: &[(&str, &str)] = &[
        ("cuda_runtime.h", "cuda-cudart-dev-<M>-<m>"),
        ("crt/host_config.h", "cuda-crt-<M>-<m>"),
        ("cublas_v2.h", "libcublas-dev-<M>-<m>"),
        ("cusolverDn.h", "libcusolver-dev-<M>-<m>"),
        ("cusparse.h", "libcusparse-dev-<M>-<m>"),
        ("nccl.h", "libnccl-dev"),
];

    let toolkit = if want_rocm {
        Some(("rocm", &rocm_path, "ROCM_PATH", "/opt/rocm", ROCM_HEADERS))
    } else if want_cuda {
        Some(("cuda", &cuda_home, "CUDA_HOME", "/usr/local/cuda", CUDA_HEADERS))
    } else {
        None
    };
    if let Some((feature, root, root_env, root_default, headers)) = toolkit {
        // Present under the toolkit root OR a default system include
        // dir -- `nccl.h` ships in libnccl-dev at /usr/include/nccl.h,
        // not under $CUDA_HOME, and checking only the toolkit root
        // reported it missing on an image that builds fine.
        // (Kept in sync by hand with flodl-cli's util/requirements.rs;
        // build.rs cannot depend on that crate.)
        let sys_dirs = ["/usr/include", "/usr/local/include"];
        let missing: Vec<&(&str, &str)> = headers
            .iter()
            .filter(|(h, _)| {
                !Path::new(root).join("include").join(h).exists()
                    && !sys_dirs.iter().any(|d| Path::new(d).join(h).exists())
            })
            .collect();
        if !missing.is_empty() {
            let mut pkgs: Vec<&str> = missing.iter().map(|(_, p)| *p).collect();
            pkgs.dedup();
            eprintln!(
                "\nflodl-sys: `--features {feature}` needs vendor toolkit headers\n\
                 that are missing under `{root}`:\n\n{}\n\n\
                 libtorch bundles the runtime libraries but not these headers.\n\n\
                 \x20 Ubuntu/Debian:  sudo apt install {}\n\
                 \x20 Other Linux:    install the vendor SDK\n\
                 \x20 macOS/Windows:  no GPU libtorch exists; on Windows use WSL2\n\n\
                 Set {root_env} if your install is not at {root_default}.\n",
                missing
                    .iter()
                    .map(|(h, p)| format!("   {h}   ({p})"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                pkgs.join(" "),
            );
            std::process::exit(1);
        }
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

        let rocm_path = ["ROCM_PATH", "HIP_PATH", "HSA_PATH"]
            .iter()
            .filter_map(|k| env::var(k).ok())
            .find(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "/opt/rocm".to_string());
        // Both layouts: `lib64` on RHEL/SUSE. A search path that does
        // not exist is ignored, so emitting both costs nothing.
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-search=native={rocm_path}/lib64");

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
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=HSA_PATH");
}
