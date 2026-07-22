---
title: "Then the CPU died"
subtitle: "flodl 0.6.0 'ddp-scale': process-per-rank multi-host DDP with an authenticated control plane, dial-in membership, elastic failure handling, and a DiLoCo outer optimizer that takes the accuracy crown. 366 commits and three months of silence, explained by one hardware failure."
date: 2026-07-22
description: "flodl 0.6.0 rebuilds the distributed layer from threads to a process-per-rank multi-host cluster: launcher, controller, coordinator, relay and agent as real processes, an HMAC-authenticated wire protocol, dial-in membership, elastic rank-death survival, and SlowMo/DiLoCo outer optimizers. On a three-GPU two-host rig, DiLoCo reaches 92.36% on CIFAR-10 ResNet-20, beating the fast GPU alone on both accuracy and wall time. This post is the honest story of why it took three months and arrived all at once."
---

flodl 0.6.0 is out, and it is by far the largest release this project
has shipped. The distributed layer went from thread-per-GPU in one
process to a full multi-host cluster: one `Trainer::run()` call now
drives a single CPU, a single GPU, N GPUs on one host, or N GPUs
across many hosts, with process-per-rank execution, an authenticated
control plane, dial-in membership, and rank-death survival. The
[changelog](https://github.com/flodl-labs/flodl/blob/main/CHANGELOG.md)
is long; the [benchmark](/ddp-benchmark) is the short version: three
mismatched GPUs spanning two hosts (one of them a VM, one card behind
a PCIe x1 riser) train ResNet-20 to **92.36%** on CIFAR-10, beating
both the published reference (91.25%) and the fast GPU alone, on
accuracy *and* wall time.

The plan for this spring was careful, incremental releases. What
follows is the honest story of why you got this instead.

## How this release actually happened

I did not plan to build this next. I was mid-arc on the meta-controller work,
running heterogeneous DDP across GPUs on a single box with threads. It worked,
and I was happily iterating on it.

Then the CPU died.

The GPUs were fine, so I kept all three, the two GTX and the RTX, and rebuilt
around a new CPU. I was still on Windows on that old rig for reasons that had
long stopped being good ones, so I took the occasion to move to Ubuntu, which is
what I already run for everything else anyway. Clean slate, better system,
overdue.

That's where the problems started. Two GPU generations in one box, Pascal and
Blackwell, and on Ubuntu no single driver would take both on the same host. The
only way back to a working setup was to give the two older cards their own VM
and pass them through. A decision made under constraints, not design.

And that quietly changed the whole game. GPUs living in a VM are GPUs living on
another machine, with a network between them. Which means no more threads. The
path I had been building on did not lose a feature. It stopped being possible on
my own hardware overnight.

Multi-host DDP was already on the roadmap. It was always where this was going.
The crash just collapsed the timeline and forced it whole. Normally you build
something this size in slices, keeping the old local path running as your daily
driver and shipping the distributed piece incrementally from a working floor.
The crash removed the floor. There was no intermediate way to run DDP at all
anymore. Rebuilding anything at all and building the full process-based,
network-aware foundation were the same job: process-per-rank execution, a
launcher, a controller, a coordinator, a relay and an agent as real processes,
an authenticated wire protocol, dial-in membership, heterogeneous cadence over
a real network, a data plane that feeds it all. None of it optional, because
DDP over a network does not exist until all of it does. All or nothing, just to
get back a capability I already had.

That's why it landed as one big release instead of a trail of small ones:
three months, 366 commits, 402 files changed, 112,000 lines added and 37,000
removed. It is also the honest reason things went quiet for a while. There was
no shippable in-between to surface. The work had to reach a floor before any of
it could stand on its own.

## Where that leaves multi-host DDP

The foundation is in, and it is solid: the machinery that makes distributed
training real rather than a demo. That is what this release is.

Multi-host DDP itself rides on top of it, and it is deliberately early. It is
weeks old and it arrived sideways. Testable, not finished. That is exactly why
I am shipping it now: to plant it as something others can run and break, and to
get back to iterating in small fast steps instead of one forced monolith. The
cloud is where this points next. The floor is rebuilt. From here it is
incremental again.

## The numbers, and where to dig

The [benchmark page](/ddp-benchmark) has the summary: 8 models, 6 cluster
modes plus 3 solo baselines, 63 runs, all on the mismatched rig described
above. Two results stand out. The whole cluster trains the 200-epoch flagship
in 8.5 minutes where the slowest card alone needs 55. And the best accuracy of
every mode tested comes from the new DiLoCo outer optimizer riding the async
path: solo training actually *reached* the same 92.36% around epoch 159, then
overfit its way back down to 92.10% over the last 40 epochs. The distributed
consensus could not follow it down: each replica only sees its own data
partition, so replica-private memorization gets averaged away every round
while shared structure survives. Distributed training as built-in early
stopping. The full report with per-epoch trajectories, per-rank schedules and
sync-cadence charts lives in
[docs/ddp-benchmark.md](https://github.com/flodl-labs/flodl/blob/main/docs/ddp-benchmark.md).

For the complete list of what landed, from the one-port mux and SSH-tunneled
workers to `fdl probe` / `fdl status` / `fdl nccl build`, see the
[changelog](https://github.com/flodl-labs/flodl/blob/main/CHANGELOG.md). To
run it: the [distributed training reference](/guide/ddp-reference) covers
everything from one GPU to a cluster, and
[`fdl setup`](/guide/cli) gets you from clone to training.

flodl remains a two-being effort: human direction, AI implementation. The
dead CPU contributed the roadmap.
