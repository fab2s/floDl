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
    if want_rocm {
        // Known-good facts, read off libtorch 2.7.0+rocm6.3's own file
        // list so P7 does not have to re-derive them:
        //   link:        torch_hip, c10_hip, amdhip64, rccl  (all four
        //                ship inside libtorch/lib, RCCL included)
        //   force-load:  libtorch_hip.so  (the analogue of the
        //                libtorch_cuda.so dlopen in ops_nn.cpp)
        //
        // What is NOT known, and is why this is an error rather than a
        // blind link: the shim includes <c10/cuda/CUDAFunctions.h>,
        // <c10/cuda/CUDACachingAllocator.h> and <nccl.h>. PyTorch
        // hipifies those paths for ROCm builds, and whether they resolve
        // verbatim against a ROCm libtorch has not been checked on
        // hardware. Linking blind would fail deep in the C++ compile
        // with a missing-header error that says nothing about the cause.
        eprintln!(
            "\nflodl-sys: the `rocm` feature is declared but the C++ shim's ROCm\n\
             path is not implemented yet.\n\n\
             The shim includes <c10/cuda/*> and <nccl.h>, which PyTorch hipifies\n\
             for ROCm builds; that has not been verified against a real ROCm\n\
             libtorch, so this build stops here rather than failing later with a\n\
             missing-header error. Build with `--features cuda`, or with no GPU\n\
             feature for CPU-only.\n"
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

    build.compile("flodl_shim");

    // Link libtorch shared libraries
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");

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
}
