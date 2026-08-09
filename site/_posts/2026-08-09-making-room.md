---
title: "Making room"
subtitle: "flodl 0.8.0 opens the framework to a second GPU vendor. AMD cards are detected, the code builds and links against ROCm, and no AMD card has run a single training step yet. This is the foundation, described honestly."
date: 2026-08-09
description: "flodl 0.8.0 adds AMD support: the GPU vendor becomes a build-time property of the process rather than an assumption baked through the codebase, GPU detection learns to read AMD's topology alongside NVIDIA's, and the public API drops CUDA from names that never meant CUDA. What is proven is that it builds, links, detects and passes its suite. What is not proven is a single kernel on real AMD silicon. This post is about the difference, and about the seven platforms flodl is now tested on."
---

flodl 0.8.0 supports a second GPU vendor. AMD cards are detected, the code
builds and links against ROCm, and the whole test suite runs against it.

No AMD card has ever run a training step.

Both sentences are true, and the second is the more useful of the two.

## What a vendor turns out to be

The library flodl builds on is compiled for exactly one backend, and its AMD
build spends its whole life pretending to be the NVIDIA one. Its tensors report
themselves as CUDA tensors. Its internal dispatch uses the CUDA identifiers.
This is the reason people with AMD cards still write `.cuda()` in PyTorch and
it simply works. The two builds cannot coexist in a single process, not because
anyone forbade it, but because both of them claim the same territory.

So the vendor is not a decision the program makes while running. It is a
property of how it was built, fixed before it starts, and the work was mostly a
matter of teaching the framework to stop assuming otherwise. One place in the
C++ layer now reconciles the two spellings, and everything above it stopped
caring. Hardware detection learned to read AMD's topology from the kernel
alongside NVIDIA's, without loading any GPU runtime to do it.

The public API changed to match. Anything named for CUDA that never actually
meant CUDA is now named for the GPU instead, with the old name kept as a
deprecated alias, so existing code keeps compiling and starts telling you what
to rename. A handful of names deliberately did not move, because the concept
underneath really is NVIDIA's and a neutral name would have been a lie that
compiles.

## What is real and what is not

Every push proves that the AMD path compiles and, more importantly, links,
because a wrong symbol mapping surfaces at link time or at the first kernel and
never during a compile. It proves the test suite passes on a machine with no
GPU driver present at all, which is not a strange edge case but exactly what a
GPU-enabled binary meets on an ordinary CPU-only box. It proves AMD cards are
found and identified correctly from a real system.

It proves nothing whatsoever about training on AMD hardware. Not one step, not
one benchmark, not one number.

There is no AMD GPU in this project's workshop. Everything above is the shape
being put in place, and the numbers that would tell you whether it is any good
do not exist yet. I would rather say that here, plainly, than let someone
discover it after an afternoon of setup. When the numbers arrive they will
arrive as numbers, not as a claim.

## Where it runs

Seven platforms now run on every push: two Ubuntu releases, Ubuntu on arm, a
Mac, Windows, and two Rocky Linux releases. Each installs the GPU toolkit the
way the documentation tells a reader to install it, natively, with no container
smoothing anything over, which means the install instructions are tested rather
than merely written down.

The Mac is new, and it is the argument for doing this at all: flodl had never
actually run on one, it compiled there perfectly well, and it failed at launch
for reasons that only running it could surface. That is fixed, a hundred
training epochs now finish in about a second and a half on a MacBook, and the
test is a hard failure rather than a warning.

## The other half

0.8.0 also finished the work that lets a machine walk into a training run
rather than being wired into it beforehand. A box can now be handed an address
and a key, prepare itself, and join a cohort already in progress, with the
controller holding the authority on what the run actually is so that a standing
fleet cannot quietly train the next experiment with the previous one's
settings.

## Foundation

Supporting more than one vendor is a means, not an achievement. The reason it
matters is that a training run can then be assembled from whatever hardware
happens to be available rather than whatever happens to be identical, and
mixed, uneven hardware is the situation this project's distributed layer was
built for.

Next step is to put flodl on real AMD silicon and find out what happens.
There is something specific I want to learn from that. For now the room
is made.

## Upgrading

Most of this release moved by rename with the old name left behind as a
deprecation, so existing code keeps working and warns you toward the new
spelling. Two things change without warning you first.

The minimum Rust version is now 1.91, pulled up by a dependency. The version
declared previously was wrong in any case, understating what the code had
already been using for months, and the build now checks the declared minimum on
every run so the two cannot drift apart again.

The HuggingFace integration needs full repository names, so
`bert-base-uncased` becomes `google-bert/bert-base-uncased`. The registry
answers the short legacy form with a redirect the updated client no longer
follows, and reports the repository as missing instead.

The [upgrade guide](https://github.com/flodl-labs/flodl/blob/main/UPGRADE.md)
covers both, and the
[changelog](https://github.com/flodl-labs/flodl/blob/main/CHANGELOG.md) covers
everything else.
