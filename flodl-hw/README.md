# flodl-hw

Hardware detection for [flodl](https://flodl.dev): GPUs and host RAM,
with **no libtorch and no CUDA runtime**. One dependency, `serde_json`,
used only to read the test-spoofing env var below.

```rust
for gpu in flodl_hw::detect_gpus() {
    println!("[{}] {} {} {} MiB", gpu.index, gpu.short_name(), gpu.arch_label(), gpu.total_memory_mb);
}
```

## Why it is its own crate

Two consumers need the same answers and cannot share code any other way:

- **`flodl`** needs GPU identity *before* libtorch is initialized. Once
  libtorch latches onto a device list, `CUDA_VISIBLE_DEVICES` is ignored,
  and on a heterogeneous rig the cluster launcher's spawned children
  inherit a corrupted CUDA context. See the "no CUDA before
  `Trainer::run`" invariant.
- **`fdl`** (`flodl-cli`) needs the same answers *before libtorch exists
  at all*, to pick which libtorch variant to download or build. It
  therefore cannot depend on `flodl`.

The two used to carry hand-synchronized copies of the same struct and the
same parser. This crate is the single source, and the place a second GPU
vendor gets added exactly once.

## Two enumerations, deliberately

| Function | Answers |
|---|---|
| `detect_gpus()` | what the **runtime** will see, visibility masks applied |
| `detect_gpus_physical()` | what is **installed**, masks ignored |

Provisioning decisions (which libtorch variant covers this box) want the
physical set. Runtime decisions (does DDP auto-promote) want the visible
set. Conflating them is a real bug in both directions, so they are named
apart rather than distinguished by a boolean.

Never initializes a GPU runtime. An absent `nvidia-smi` is "no GPUs",
not an error.

## Spoofing hardware

`FLODL_TESTING_GPU_JSON` replaces the whole sweep, so detection and
downstream routing can be tested on a machine without the hardware:

```bash
FLODL_TESTING_GPU_JSON='[{"vendor":"amd","arch":"gfx1030","vram_mb":16384}]' \
  cargo test
```

Visibility masks still apply on top. A malformed value panics rather
than falling back to real hardware, so a typo cannot silently make a
test report on the wrong machine.

## License

MIT
