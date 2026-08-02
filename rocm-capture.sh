#!/usr/bin/env bash
# Dump everything a ROCm host can tell us about itself, into one tarball.
#
# Written for a rented AMD box whose clock is running: the machine goes
# away, so anything not captured has to be re-rented to learn. It is also
# a general support dump -- run it on any host and attach the tarball to a
# bug report.
#
#   ./rocm-capture.sh [output-dir]
#
# No dependencies beyond coreutils. Every probe is optional: a missing
# tool records why it is missing instead of aborting the run, because the
# absence of `amd-smi` is itself a finding. Safe to run as an unprivileged
# user; the few root-only reads degrade to a note.
#
# Run it in BOTH contexts on a containerised host (once on the host, once
# inside the container). The device-node layout differs between them, and
# that difference is the thing several of our checks reason about.

set -u

STAMP=$(date -u +%Y-%m-%dT%H-%M-%SZ)
HOST=$(hostname 2>/dev/null || echo unknown-host)
OUT_ROOT=${1:-.}
OUT="$OUT_ROOT/rocm-capture-$HOST-$STAMP"
mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 1; }

# Bounded, so a cold ROCm runtime cannot hang the capture. amd-smi's first
# call initialises the whole stack and is the slow one.
TIMEOUT=${ROCM_CAPTURE_TIMEOUT:-30}

note() { printf '%s\n' "$*" | tee -a "$OUT/capture.log"; }

# run <file> <command...> -- capture stdout+stderr and the exit code.
# A command that is not installed is recorded as such rather than as a
# failure: "not installed" and "installed but broken" are different facts.
run() {
    local file="$OUT/$1"; shift
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'unavailable: %s is not on PATH\n' "$1" > "$file"
        return
    fi
    { timeout "$TIMEOUT" "$@" ; printf '\n[exit %s]\n' "$?" ; } > "$file" 2>&1
}

# copy_tree <dest> <glob...> -- concatenate readable /sys or /proc files,
# each under a header naming its path.
copy_tree() {
    local file="$OUT/$1"; shift
    : > "$file"
    local found=0 p
    for p in "$@"; do
        [ -e "$p" ] || continue
        found=1
        printf '===== %s =====\n' "$p" >> "$file"
        if [ -r "$p" ]; then
            cat "$p" >> "$file" 2>&1
        else
            printf '(unreadable: %s)\n' "$(ls -ld "$p" 2>&1)" >> "$file"
        fi
        printf '\n' >> "$file"
    done
    [ "$found" -eq 1 ] || printf 'unavailable: no path matched\n' > "$file"
}

note "rocm-capture $STAMP on $HOST -> $OUT"

# --- 0. execution context ---------------------------------------------
# Which side of the container boundary this run describes. Several checks
# (uvm presence, /dev/dri passthrough) only mean something paired with it.
{
    printf 'captured_at=%s\n' "$STAMP"
    printf 'hostname=%s\n' "$HOST"
    printf 'whoami=%s\n' "$(id -un 2>/dev/null)"
    printf 'uid=%s gid=%s\n' "$(id -u 2>/dev/null)" "$(id -g 2>/dev/null)"
    printf 'groups=%s\n' "$(id -nG 2>/dev/null)"
    printf 'pwd=%s\n' "$PWD"
    if [ -f /.dockerenv ]; then
        printf 'container=yes (/.dockerenv present)\n'
    elif grep -qE '(docker|containerd|kubepods|libpod)' /proc/1/cgroup 2>/dev/null; then
        printf 'container=likely (cgroup match on pid 1)\n'
    else
        printf 'container=no\n'
    fi
    # systemd-detect-virt exits non-zero when it finds nothing, so the
    # `|| true` is what keeps a bare-metal reading from printing twice.
    printf 'virt=%s\n' "$(systemd-detect-virt 2>/dev/null || true)"
} > "$OUT/00-context.txt"

# --- 1. machine --------------------------------------------------------
run 01-uname.txt uname -a
copy_tree 01-os-release.txt /etc/os-release
copy_tree 01-meminfo.txt /proc/meminfo
copy_tree 01-cpuinfo.txt /proc/cpuinfo
run 01-nproc.txt nproc
run 01-lsblk.txt lsblk
run 01-df.txt df -h
run 01-lscpu.txt lscpu
copy_tree 01-cmdline.txt /proc/cmdline

# --- 2. AMD userspace stack -------------------------------------------
# amd-smi output is THE artifact this capture exists for: the marketing-name
# parser and ROCm-version resolution are blocked on never having seen real
# output. Take every subcommand, including the ones that may not exist.
run 02-amd-smi-version.txt amd-smi version
run 02-amd-smi-list.txt amd-smi list
run 02-amd-smi-static.txt amd-smi static
run 02-amd-smi-static-asic.txt amd-smi static --asic
run 02-amd-smi-monitor.txt amd-smi monitor
run 02-amd-smi-list-json.txt amd-smi list --json
run 02-amd-smi-static-json.txt amd-smi static --json
run 02-amd-smi-help.txt amd-smi --help
run 02-rocminfo.txt rocminfo
run 02-rocm-smi.txt rocm-smi
run 02-rocm-smi-driver.txt rocm-smi --showdriverversion
run 02-rocm-smi-product.txt rocm-smi --showproductname
run 02-rocm-smi-mem.txt rocm-smi --showmeminfo vram
run 02-hipconfig.txt hipconfig --full
run 02-hipcc-version.txt hipcc --version

# ROCm version, every published source, because they disagree.
copy_tree 02-rocm-version.txt \
    /opt/rocm/.info/version /opt/rocm/.info/version-dev \
    /opt/rocm/.info/version-libs /opt/rocm/.info/version-utils
run 02-dpkg-rocm.txt dpkg-query -W -f '${Package} ${Version}\n' 'rocm*' 'hip*' 'hsa*'
run 02-rpm-rocm.txt rpm -qa 'rocm*'

# The libraries D1a's LD_LIBRARY_PATH ordering puts ahead of libtorch's
# bundle. A version disagreement here is the segfault-at-first-op case.
run 02-ls-opt-rocm.txt ls -l /opt/rocm
run 02-ls-opt-rocm-lib.txt ls -l /opt/rocm/lib
run 02-ls-opt-rocm-lib64.txt ls -l /opt/rocm/lib64
{
    printf 'ROCM_PATH=%s\n' "${ROCM_PATH:-<unset>}"
    printf 'HIP_PATH=%s\n' "${HIP_PATH:-<unset>}"
    printf 'HSA_PATH=%s\n' "${HSA_PATH:-<unset>}"
    for root in "${ROCM_PATH:-}" "${HIP_PATH:-}" "${HSA_PATH:-}" /opt/rocm; do
        [ -n "$root" ] || continue
        for sub in lib lib64; do
            for so in libhsa-runtime64.so libhsa-runtime64.so.1; do
                [ -e "$root/$sub/$so" ] && printf 'hsa-runtime: %s/%s/%s\n' "$root" "$sub" "$so"
            done
        done
    done
} > "$OUT/02-hsa-runtime-candidates.txt" 2>&1

# --- 3. kernel driver + KFD topology ----------------------------------
# The KFD tree is the mask-proof detection substrate: vendor_id 4098 is
# an AMD GPU node no visibility variable can hide. Take the whole tree --
# a node's `name` file does NOT hold the gfx name, so the decode depends
# on fields we want to re-read at home.
copy_tree 03-amdgpu-version.txt /sys/module/amdgpu/version
run 03-lsmod.txt lsmod
copy_tree 03-kfd-topology-props.txt /sys/class/kfd/kfd/topology/nodes/*/properties
copy_tree 03-kfd-topology-name.txt /sys/class/kfd/kfd/topology/nodes/*/name
copy_tree 03-kfd-topology-gpuid.txt /sys/class/kfd/kfd/topology/nodes/*/gpu_id
run 03-kfd-tree.txt find /sys/class/kfd/kfd/topology -maxdepth 3
copy_tree 03-kfd-system-props.txt /sys/class/kfd/kfd/topology/system_properties
if dmesg >/dev/null 2>&1; then
    dmesg 2>/dev/null | grep -iE 'amdgpu|kfd|hsa' > "$OUT/03-dmesg-amdgpu.txt" 2>&1
else
    printf 'unavailable: dmesg is restricted (kernel.dmesg_restrict) or absent\n' \
        > "$OUT/03-dmesg-amdgpu.txt"
fi

# --- 4. device nodes and permissions ----------------------------------
# The layout our permission check reasons about, on a machine that is not
# the dev box. Both the AMD pair (/dev/kfd + /dev/dri/renderD*) and the
# NVIDIA mirror (/dev/nvidia-uvm, which nvidia-smi does not need but CUDA
# init does, so a container missing it looks healthy and dies at init).
{
    for n in /dev/kfd /dev/dri /dev/dri/renderD* /dev/dri/card* \
             /dev/nvidia* /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
        if [ -e "$n" ]; then
            ls -ld "$n" 2>&1
        else
            printf 'ABSENT   %s\n' "$n"
        fi
    done
} > "$OUT/04-device-nodes.txt" 2>&1
run 04-ls-dev-dri.txt ls -lR /dev/dri
copy_tree 04-proc-self-status.txt /proc/self/status
run 04-id.txt id
run 04-getent-groups.txt getent group render video

# Can this user actually OPEN the nodes? Presence plus group membership is
# not proof; an unopenable /dev/kfd is not a usable GPU.
{
    for n in /dev/kfd /dev/dri/renderD128 /dev/nvidiactl /dev/nvidia-uvm; do
        if [ ! -e "$n" ]; then
            printf '%-24s ABSENT\n' "$n"
        elif : < "$n" 2>/dev/null; then
            printf '%-24s OPENABLE\n' "$n"
        else
            printf '%-24s PRESENT BUT NOT OPENABLE\n' "$n"
        fi
    done
} > "$OUT/04-node-openability.txt" 2>&1

# --- 5. PCI ------------------------------------------------------------
# Sharpening for the case where ROCm cannot use the card at all: an AMD
# display-class device exists on the bus whether or not a runtime is there.
run 05-lspci.txt lspci -nnv
{
    for d in /sys/bus/pci/devices/*; do
        [ -r "$d/vendor" ] || continue
        v=$(cat "$d/vendor" 2>/dev/null)
        c=$(cat "$d/class" 2>/dev/null)
        case "$v" in
            0x1002|0x10de)
                printf '%s vendor=%s class=%s device=%s driver=%s\n' \
                    "$(basename "$d")" "$v" "$c" \
                    "$(cat "$d/device" 2>/dev/null)" \
                    "$(basename "$(readlink "$d/driver" 2>/dev/null)" 2>/dev/null)"
                ;;
        esac
    done
} > "$OUT/05-pci-gpu-vendors.txt" 2>&1

# --- 6. NVIDIA leg -----------------------------------------------------
# Captured on an AMD box too: "no NVIDIA anything" is the expected reading
# and confirms the box is what we think it is.
run 06-nvidia-smi.txt nvidia-smi
run 06-nvidia-smi-query.txt nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
copy_tree 06-nvidia-version.txt /proc/driver/nvidia/version

# --- 7. environment ----------------------------------------------------
# Masks first: HIP honours HIP_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES and
# CUDA_VISIBLE_DEVICES, in that precedence. A container mask here changes
# what every other tool in this dump reported.
{
    env | grep -E '^(ROCM|ROCR|HIP|HSA|GPU|AMD|CUDA|NCCL|RCCL|PYTORCH|LD_|LIBTORCH|FDL|FLODL)' | sort
    printf '\n--- masks, explicitly ---\n'
    for v in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
        printf '%s=%s\n' "$v" "${!v-<unset>}"
    done
} > "$OUT/07-env.txt" 2>&1

# --- 8. python torch, if the image ships one --------------------------
# Answers whether the box already carries a libtorch we could point at
# (wheels bundle the C++ libs under torch/lib), and what ROCm it was built
# for. torch.version.hip non-empty is the canonical "this is a ROCm stack".
for py in python3 python; do
    command -v "$py" >/dev/null 2>&1 || continue
    timeout "$TIMEOUT" "$py" - <<'PY' > "$OUT/08-torch.txt" 2>&1
import json
try:
    import torch
except Exception as e:
    print("no torch:", e)
    raise SystemExit(0)
info = {
    "version": torch.__version__,
    "file": torch.__file__,
    "version.hip": getattr(torch.version, "hip", None),
    "version.cuda": getattr(torch.version, "cuda", None),
}
try:
    info["cuda.is_available"] = torch.cuda.is_available()
    info["device_count"] = torch.cuda.device_count()
    info["arch_list"] = torch.cuda.get_arch_list()
    if torch.cuda.device_count():
        p = torch.cuda.get_device_properties(0)
        info["props0"] = {
            "name": p.name,
            "gcnArchName": getattr(p, "gcnArchName", None),
            "integrated": getattr(p, "integrated", None),
            "warp_size": getattr(p, "warp_size", None),
            "total_memory": p.total_memory,
            "major": p.major, "minor": p.minor,
        }
except Exception as e:
    info["probe_error"] = repr(e)
print(json.dumps(info, indent=2, default=str))
PY
    break
done
[ -f "$OUT/08-torch.txt" ] || printf 'unavailable: no python on PATH\n' > "$OUT/08-torch.txt"

# --- 9. flodl, if this box already has it -----------------------------
# Runs late on purpose: it is the only section that needs our own build,
# and the capture must be useful before one exists.
if command -v fdl >/dev/null 2>&1; then
    run 09-fdl-probe.txt fdl probe
    run 09-fdl-probe-json.txt fdl probe --json
    run 09-fdl-version.txt fdl --version
else
    printf 'unavailable: fdl is not on PATH (capture ran before install)\n' \
        > "$OUT/09-fdl-probe.txt"
fi
# rocblas ships a machine-readable arch manifest: which gfx targets this
# libtorch actually carries kernels for.
for lt in "${LIBTORCH_PATH:-}" ./libtorch/precompiled/rocm70 /opt/libtorch; do
    [ -n "$lt" ] && [ -d "$lt/lib/rocblas/library" ] || continue
    ls "$lt/lib/rocblas/library" > "$OUT/09-rocblas-arch-manifest.txt" 2>&1
    printf '(from %s)\n' "$lt" >> "$OUT/09-rocblas-arch-manifest.txt"
    break
done

# --- pack --------------------------------------------------------------
TARBALL="$OUT.tar.gz"
if tar czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null; then
    note "packed: $TARBALL ($(du -h "$TARBALL" 2>/dev/null | cut -f1))"
else
    note "tar failed; the directory $OUT is still complete"
fi

note "sections written: $(find "$OUT" -type f | wc -l)"
note "COPY THE TARBALL OFF THE BOX BEFORE ANYTHING ELSE."
