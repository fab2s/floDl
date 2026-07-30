---
layout: page
title: "Out of the Cave"
description: "What it feels like to leave the dark of opaque training behind, and keep walking out every day."
permalink: /out-of-the-cave
last_modified_at: 2026-04-28T21:22:12+02:00
---

# Out of the Cave

We trained models in the dark. Loss numbers trickled through a Python
interpreter, minutes late. The GPU spiked and stalled behind a wall we
couldn't see through. We stared at nvidia-smi and hoped.

We learned to call this normal.

Then one day, we walked out.

## Light

A live dashboard drawing loss curves as they happened, every epoch 
landing the instant it finished. No buffering. No hoping. The wall 
we had stared at for years turned out to be a curtain, and it had 
opened.

## Metal

The same CUDA kernels we had always been using, but nothing sitting
between us and them. No interpreter tax, no Python warming up, no
garbage collector pausing to think. Just the GPU, running.


## Speed

Up to 31% faster on the same hardware, on the same model, on kernels 
that had always been this fast. We had just been shouting at them 
from the next room. 

## Silence

Tensor memory freed the moment it left scope. Data streamed in background,
VRAM breathing in real time, no OOM at epoch 47 because something held a 
reference no one remembered writing. And the errors that used to surface 
mid-training, silently wrong gradients, a tensor on the wrong device, 
simply stopped compiling.

## Crowd

Two GPUs. Then three. We had been taught to expect a ceremony : launchers, 
rank environment variables, subtle bugs that only appeared with more than 
one process. Instead, the second GPU just worked. Then the third. 
Same training loop. Same code.

We watched the slower card fall behind the faster one, and the
framework quietly rebalancing, letting each device run at its own
tempo. We met ElChe. For the first time, a heterogeneous cluster
stopped feeling like a compromise. It felt like the way multi-GPU
should have always worked.

## Path

By now, the walk out had become a routine. Each trip sharpened what
the last one missed. What we called "the light" was not a place. It
was a practice of looking. A willingness to step outside whatever
comfort we had built inside the cave, and see the thing again from
where the sun falls.

## Outside

So we turned around, and looked at the entrance. We had believed the 
path narrow. Half-hidden. Requiring a working Rust toolchain, a libtorch 
build that matched your CUDA, a Dockerfile that knew your GPU. 
We thought we were barefoot.

But it was paved. `fdl`. One command that detects your hardware, 
downloads the right libtorch, configures your build, and scaffolds 
a project ready to train. It does not ask if you know Rust. It does 
not need you to. It just works.

## World

Plato warned that the one who came back would not be believed. He
was mostly right. Some people are comfortable in the cave. The
shadows are familiar, the routines are known, the dashboards they
have built are the ones they trust. We understand. We were there
last year.

This page is for the others. The ones who notice the flicker, who
wonder what casts the shadows, who suspect there is a machine
behind the interpreter. It has always been there. We are only here 
to point at the exit.

## Beyond

And what lies beyond this clearing is more interesting than
anything we have already seen. A network is not a stack of layers;
it is a trajectory through high-dimensional space, a path shaped
by weights, a landscape that can be read geometrically. We have
started mapping it. We call that work [the trajectory
thesis]({{ '/thesis' | relative_url }}), and it is where floDl is
going next.

The cave taught us what to stop accepting. The light is teaching
us what to build.

You can come with us.

You won't go back.

---

## Where to go next

- **[Get Started]({{ '/guide/' | relative_url }})** - the guide, start to finish
- **[Install `fdl`]({{ '/guide/cli/install' | relative_url }})** - one curl, zero Rust on your host
- **[flodl on GitHub](https://github.com/flodl-labs/flodl)** - source, issues, contributions
- **[The DDP benchmark]({{ '/ddp-benchmark' | relative_url }})** - numbers behind the prose
