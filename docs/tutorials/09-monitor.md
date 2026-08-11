# Training Monitor

The training monitor provides human-readable progress output, system resource
tracking, and an optional live web dashboard - all with zero external
dependencies.

> **Prerequisites**: [Training](04-training.md) covers the training loop.
> [Utilities](08-utilities.md) covers observation and trends.

> **Runnable examples**: [`quickstart`](../../flodl/examples/quickstart/) trains with
> a live monitor; [`observation`](../../flodl/examples/observation/) adds trend queries
> and early stopping.

## Basic Usage

The monitor wraps your training loop with timing, ETA, and resource sampling.
Record metrics on the graph during training, then pass the graph to `log`:

```rust
use flodl::*;
use flodl::Monitor;

let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(Linear::new(16, 2)?)
    .build()?;

let params = model.parameters();
let mut optimizer = Adam::new(&params, 0.01);
model.train();

let num_epochs = 100;
let mut monitor = Monitor::new(num_epochs);

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();

    for (input_t, target_t) in &batches {
        let input = Variable::new(input_t.clone(), true);
        let target = Variable::new(target_t.clone(), false);
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;

        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;

        model.record_scalar("loss", loss.item()?);
    }

    model.flush(&[]);
    monitor.log(epoch, t.elapsed(), &model);
}

monitor.finish();
```

Each `log` call prints a one-liner to stderr:

```
  epoch   1/100  loss=1.5264  [49ms  ETA 4.8s]
  epoch   2/100  loss=1.1020  [28ms  ETA 3.6s]
  epoch  50/100  loss=0.0023  [24ms  ETA 1.2s]  VRAM: 2.1/6.0 GB (82%)
  epoch 100/100  loss=0.0012  [23ms]
  training complete in 2.8s  | loss: 0.0012
```

The ETA adapts its format automatically: `3h 12m`, `4m 32s`, `12s`, `420ms`.

GPU metrics appear automatically when a GPU is available, and are silently
omitted on CPU-only builds. VRAM works on either vendor; utilization is
NVIDIA-only today (see [Resource Tracking](#resource-tracking)).

## Multiple Metrics

Record everything on the graph - `flush` averages each tag independently:

```rust
model.record_scalar("loss", loss.item()?);
model.record_scalar("grad_norm", norm);
model.record_scalar("lr", scheduler.lr(epoch));
model.flush(&[]);
monitor.log(epoch, t.elapsed(), &model);
// epoch  42/100  loss=0.0023  grad_norm=0.4521  lr=0.0008  [1.2s  ETA 1m 10s]
```

You can also pass metrics manually without a graph:

```rust
monitor.log(epoch, t.elapsed(), &[("loss", avg_loss), ("lr", lr)]);
```

## Live Dashboard

Start an embedded HTTP server to get a real-time web dashboard:

```rust
let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;  // http://localhost:3000
```

Open `http://localhost:3000` in a browser. The dashboard shows:

- **Header**: epoch counter, progress bar, ETA, elapsed time
- **Breadcrumb**: which level you are looking at, and the way back up
- **Metrics chart**: live-updating canvas chart of this level's metrics
- **Resource chart**: GPU% and VRAM% over time (plus CPU/RAM at the run level)
- **Resource bars**: current values with percentage fill
- **Children**: the level below, compared on one metric and drillable
- **Log table**: this level's rows, newest first
- **Alerts**: rank losses, drift, dropped control frames — whole run, at any level
- **Graph SVG**: collapsible architecture diagram (if provided)

### One view, repeated per level

The page is a **portal**: every level renders the same way, and the
breadcrumb is the record `path`. A single-GPU run is just the run level.
A cluster run starts at `root` and drills down — click a child row to
descend, a breadcrumb segment to come back:

```
root                          the cohort: work-weighted roll-up of every host
root/flodl-pascal             one host: roll-up of the ranks on it
root/flodl-pascal/rank1       one rank: its own measurements, nothing averaged
```

Each level is linkable (`#path=root/flodl-pascal/rank1`) and the browser's
Back button walks the levels.

Two details make the levels readable:

- **The legend says what it is showing.** At an interior level a metric is a
  roll-up over the direct children, so the legend names the reduction
  (`loss (mean)`, `throughput (sum)`); at a leaf it is a raw measurement and
  the legend is the bare key. Children are named by their path segment plus
  the label the producer attached, e.g. `rank1 · GTX 1060 6GB`.
- **`work` is a column, never a curve.** It is a per-record interval quantity
  whose unit differs by cadence (steps for a sub-epoch window, batch share for
  an epoch boundary), so plotting the two as one series would compare
  different units.

Host CPU and RAM are per-host facts that deliberately do not live in the
record tree (summing co-hosted ranks would double-count them). They show as
gauges — cohort mean / cohort total in a cluster run — rather than as a curve
on an axis they do not share.

### How it works

The server uses raw TCP sockets and [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-Sent_Events) (SSE) - no HTTP framework, no WebSocket library, no JavaScript dependencies.

1. `monitor.serve(port)` spawns a background listener thread
2. Each browser connection gets its own handler thread
3. `GET /` serves the dashboard HTML (all JS/CSS inline, ~53 KB)
4. `GET /events` holds the connection open as an SSE stream
5. Each `monitor.log(...)` pushes a JSON event to all connected clients
6. The browser JS updates the charts and log in real time

`/events` is the **run clock**: epoch counter, ETA, elapsed, completion, and
the host resource gauges. The levels come from the path-scoped feed alongside
it — see [Querying the run by path](#querying-the-run-by-path). A run without
a record plane (any single-process run) has no levels to browse, so the page
builds them from the epoch feed instead: the run level, plus one child per
device when the run reports two or more. Same renderer either way — and the
same applies to a [saved archive](#dashboard-archive), which carries whichever
of the two the run produced.

### Late join

If you open the dashboard mid-training, it catches up instantly. The SSE
handler replays all past epoch events before switching to live streaming.

### Light and dark

The toggle sits at the top right of the header. Your choice is remembered per
browser and always wins from then on.

Unset, the page follows your OS via `prefers-color-scheme` — so a dark desktop
gets the dark dashboard, which is the better place to watch a long run, and a
light desktop gets the light one without having to ask. The palette is the same
one flodl.dev uses, under the same variable names.

The charts follow too. Canvas has no CSS cascade, so the chart colours are
resolved from the stylesheet on each render rather than hardcoded, and the
series palette swaps for a darker ramp in light mode — the pale end of the dark
ramp is unreadable on white.

### Multi-GPU and cluster runs

`monitor.serve(port)` works the same way on single-host multi-GPU and
multi-host clusters: **one URL covers the whole run**. When the
launcher fans out to multiple rank processes (auto-promoted on 2+
GPUs, or via `fdl @cluster <cmd>`), the dashboard grows the levels
described above — hosts under `root`, ranks under each host, each with
throughput, batch share, VRAM and GPU utilization. No extra wiring: open
`http://<launcher-host>:3000` and drill down.

The page subscribes to the level you are on, so a 3-rank rig and a
300-rank rig cost the same to watch. It keeps one extra subscription
pinned at `root`, because alerts are scoped to the whole subtree there —
a rank loss deep in an unwatched branch still reaches you.

### Sub-epoch reports

The curves above plot one point per **epoch**. When an epoch is long — or
when there is only one (single-pass LLM training) — ask the controller for
intermediate points:

```rust
Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .num_epochs(1)
    .reports_per_epoch(20)   // ~20 loss points across the run
    .run()?;
```

Reports fire at reduce boundaries, up to `n` per epoch, carrying per-rank
loss / throughput aggregated up a `root → host → rank` tree. Off by
default, and the per-epoch feed is unchanged either way. Details:
[DDP guide → Sub-epoch reports](../ddp/01-reference.md#sub-epoch-reports---reports_per_epoch).

### Querying the run by path

Those reports also land in a **path-addressable record plane** on the same
port, so you can read any level of the cluster without a browser:

```
GET /paths                       # every node path currently reporting
GET /node?path=root              # one level: this node + its direct children
GET /node?path=root/exa/rank0    # drill in — same shape at any depth
GET /history?path=root&n=200     # the last N records for that level
GET /stream?path=root/exa        # SSE, live, scoped to that level
```

```console
$ curl -s localhost:3000/paths
["root","root/exa","root/exa/rank0","root/pascal","root/pascal/rank1"]
```

Two properties are worth knowing, because they are what make this usable at
any scale:

- **A level costs `O(children)`, never `O(cluster)`.** `/node` answers with
  the node's own aggregate plus one record per *direct* child — the same
  "aggregate over direct children only" rule that lets the tree render
  identically at every depth. A root query on a 1000-rank run returns as much
  data as one on a 3-rank run.
- **`/history` returns exactly what `/stream` would have sent.** They share
  one scoping rule, so "read the history, then subscribe" has no gap and no
  duplicate at the handover.

Metrics are scoped to depth 1 (a level renders from its direct children), but
**alerts are not**: a `rank_lost` deep in the tree reaches a `root` subscriber
too. You should not have to be looking at the right level to find out a rank
died.

#### Two cadences, one stream per level

Every level's stream carries **both** report cadences, interleaved:

| | sub-epoch window report | epoch-boundary record |
|---|---|---|
| when | at reduce boundaries, up to `reports_per_epoch` | once per epoch |
| marked | `tick: <n>` | `epoch_complete: true` |
| carries | loss, throughput, compute_only_ms, batch_share, `res` when freshly sampled | all of that **plus** user scalars (`record_scalar`, eval metrics) |

So the dense curve comes from the window reports and the epoch rows punctuate
it — `jq 'select(.epoch_complete)'` gives you just the epoch series.

Resources (`gpu_util` / `vram_alloc` / `vram_total`) are sampled on their own
~500 ms cadence, independent of both, so a **single-epoch** run gets a GPU/VRAM
curve rather than one reading for the whole run. A record carries `res` only
when a fresh sample arrived since the last one reported: repeating the previous
value would smear one reading across the epoch, so absent stays absent instead.

`GET /node?path=<p>` folds the two together into the node's **current state** —
each field showing its most recent reading — so a gauge does not blink off
between epochs. `/history` and `/stream` keep them as separate rows, which is
what makes a level's log readable.

Nodes also carry an optional `label` (a rank's GPU model), so a legend at any
level can read `rank0 · RTX 5060 Ti` instead of just `rank0`.

One caveat: `work` is a **per-record interval** quantity, and its unit differs
by cadence (steps for a window, batch-share for an epoch). Every rollup is
exact within a record — it is the aggregation weight — but do not plot it as a
single curve across both.

### Embedding the graph

To show the graph architecture in the dashboard:

```rust
let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;
monitor.watch(&model);  // generates SVG, sends to dashboard
```

The SVG appears in a collapsible section at the bottom of the dashboard. It is
**run-scoped, not level-scoped**: the architecture describes the whole run, so
the portal shows it at the root level only and never repeats it per host or per
rank. The same holds for the "Training Configuration" card fed by
`set_metadata`.

A deep model's graph is a tall, narrow ribbon (ResNet-56 renders roughly 272 ×
7500 points), so the card renders it at natural size and scrolls internally
rather than scaling it to the card width.

`watch` also derives the parameter counts (`total` / `trainable` / `frozen`)
and publishes them into the same configuration card, so this works with no
`set_metadata` call at all. When you do call both, order does not matter and
your own keys win on collision:

```rust
monitor.set_metadata(serde_json::json!({ "lr": 0.1, "seed": 42 }));
monitor.watch(&model);   // adds `parameters`, keeps `lr` / `seed`
```

Graphviz (`dot`) must be installed for the SVG; when it is missing, `watch`
still publishes the parameter counts and simply omits the drawing.

For a timing-annotated heat map (green/yellow/red by execution time), enable
profiling during training and use `finish_with`:

```rust
model.enable_profiling();

for epoch in 0..num_epochs {
    // ... training (profiling records timing on each forward pass) ...
}

monitor.finish_with(&model);  // final SVG with steady-state timing heat map
```

`finish_with` generates the profiled SVG at the end of training - when the
last forward pass timing is representative of steady-state performance. The
heat map is pushed to the live dashboard and baked into the HTML archive.

You can also update the SVG mid-training with `watch_profiled(&model)`.

Both methods require Graphviz (`dot`) to be installed. If `dot` is not
available, they silently fall back or do nothing.

On cluster runs the heat map needs no `dot` anywhere and no monitor calls:
set `TrainerConfig::profile_graph` and every rank profiles its training
graph with device-side events, shipping accumulated per-node min/mean to
the controller at teardown. The dashboard's Graph Architecture card gains
one heat map per GPU model (averaging across different models would
describe no device that exists), with mean/min in each node's hover
tooltip and a legend carrying the clock provenance. The SVG download
button saves exactly the bytes on screen, publication-ready; the saved
HTML archive embeds the same finished artifacts.

## Resource Tracking

The monitor samples system resources on every `log` call:

| Metric | Source | When available |
|--------|--------|----------------|
| CPU % | `/proc/stat` (delta) | Linux |
| RAM used/total | `/proc/meminfo` | Linux |
| GPU utilization % | NVML (dynamic load) | NVIDIA GPU + driver |
| VRAM allocated / spill | libtorch caching allocator (`reserved_bytes`) | GPU feature enabled |

Resources that aren't available are silently omitted from both the terminal
output and the dashboard.

**On ROCm, utilization is the one gap.** The allocator-backed VRAM figures
come from libtorch and work on either vendor, but the utilization probe
loads NVML, which is NVIDIA-only: on an AMD box it fails to load and the
metric is omitted rather than reported wrong. Everything else in the
dashboard — throughput, VRAM, batch share, timings — is unaffected. Use
`rocm-smi` alongside the run for utilization until an AMD SMI probe lands.

### VRAM metrics

flodl exposes two levels of GPU memory measurement. Both read libtorch's
caching allocator, so both work on either vendor (the `gpu_` names are
vendor-neutral for that reason; ROCm keeps the CUDA device type all the
way down):

| Function | What it measures | PyTorch equivalent |
|----------|-----------------|-------------------|
| `gpu_active_bytes()` | Bytes backing live tensors | `torch.cuda.memory_allocated()` |
| `gpu_allocated_bytes()` | Total allocator reservation (includes cached free blocks) | `torch.cuda.memory_reserved()` |

The monitor tracks `gpu_allocated_bytes` (reserved) because it detects
unified-memory spill - when reserved bytes exceed physical VRAM, the
allocator has spilled to host RAM.

For debugging, compare both: if `active` is small but `reserved` is large,
the allocator is holding freed blocks. Call `gpu_empty_cache()` to release them.

### Accessing resource data

```rust
for record in monitor.history() {
    if let Some(alloc) = record.resources.vram_allocated_bytes {
        println!("epoch {}: VRAM {} bytes", record.epoch, alloc);
    }
}
```

## Export

### Dashboard archive

Save the full dashboard as a self-contained HTML file - all charts, resource
graphs, epoch log, and graph SVG baked in. Open it in any browser, no server,
no sibling files.

```rust
monitor.save_html("training_report.html");  // set before training
// ... training loop ...
monitor.finish();  // writes the archive
```

The archive is written automatically when `finish()` is called. It is the same
dashboard you saw live, frozen at the final state — including the **record
plane**, so a saved cluster run is the full portal: every level browsable, both
metric cadences interleaved, `#path=` deep links working. It is not a flattened
screenshot of the root view.

**It cannot grow without bound.** The record plane is a ring
(`record_store::MAX_RECORDS`), so a long run shortens the archive's *horizon*
rather than enlarging its *file* — which is what keeps it one attachable
artifact regardless of how long you trained.

#### On a cluster run

Ask for it through the builder, not through your own `Monitor`:

```rust
Trainer::builder(model, opt, step)
    .save_dashboard("runs/exp1/dashboard.html")
```

Your `Monitor` has neither the dashboard server nor the records on a cluster run
— `serve()` returns early there and the launcher's internal sink owns both — so
`monitor.save_html(...)` would write a page with no levels and no curves. The
builder routes the request to the sink that has the data. `ddp-bench` exposes
this as `--save-dashboard`.

Either way it needs **no dashboard port**: persisting a dashboard does not
require serving one.

#### Theme, and the publication case

A saved page follows the reader's OS by default, exactly as the live one does.
Pin it when the artifact is headed somewhere with a fixed look — a figure in a
paper should not change appearance with the reviewer's desktop:

```rust
Trainer::builder(model, opt, step)
    .save_dashboard("runs/exp1/dashboard.html")
    .dashboard_theme("light")          // "light" | "dark" | "auto"
```

`ddp-bench` exposes it as `--dashboard-theme light`.

You do not have to decide at training time. Every saved page carries the choice
as a single line near the top, so re-theming an artifact you already have is one
edit:

```js
const ARCHIVE_THEME=null;      // null = follow the reader's OS
                               // "light" to pin it for publication
```

The reader's own toggle still overrides whatever is pinned — pinning sets the
*default*, it does not take the control away.

### Training log

```rust
monitor.write_log("training.log")?;
```

Produces:

```
# flodl training log
epoch   1/100  loss=1.5264  [49ms]
epoch   2/100  loss=1.1020  [28ms]
...
# total: 2.8s
```

### CSV

```rust
monitor.export_csv("training.csv")?;
```

Produces:

```csv
epoch,duration_s,loss,cpu_pct,ram_used,gpu_pct,vram_alloc,vram_spill
1,0.049,1.5264,45.2,3221225472,82.0,2254857830,0
2,0.028,1.1020,43.8,3221225472,81.5,2254857830,0
...
```

## Hierarchical Models (Graph Tree)

When a graph has labeled children (see [Graph Tree](10-graph-tree.md)),
`flush()` and `latest_metrics()` are tree-aware. A single flush on the parent
propagates to all children, and the monitor automatically sees child metrics
with dotted prefixes:

```rust
let subscan = FlowBuilder::from(scan_module)
    .label("subscan")
    .build()?;
let letter = FlowBuilder::from(letter_module)
    .label("letter")
    .build()?;
let model = FlowBuilder::from(subscan)
    .through(letter)
    .build()?;

let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();

    for batch in &batches {
        // ... forward, backward, step ...
        model.record_at("subscan.ce", ce_value)?;
        model.record_at("letter.accuracy", acc)?;
        model.record_scalar("total_loss", total);
    }

    model.flush(&[]);  // flushes parent + subscan + letter
    monitor.log(epoch, t.elapsed(), &model);
    // Output: epoch 1/100  total_loss=0.42  subscan.ce=0.31  letter.accuracy=0.87  [1.2s ETA 2m]
}
```

The dashboard shows each metric as a separate curve. Dotted names group
naturally in the legend - you can solo-click `subscan.ce` to focus on it.

If child subgraphs flush on a different cadence, use `flush_local()` to manage
them independently. See [Independent flush cadences](10-graph-tree.md#independent-flush-cadences).

## Monitor vs. Graph Observation

floDl has two metric systems that serve different purposes:

- **Graph observation** (`record`/`flush`/`trend`) - metrics that **feed back
  into training**. Use trends to trigger early stopping, LR decay, or
  convergence checks. The graph owns this data and your training loop reads it.

- **Monitor** (`log`/`serve`/`save_html`) - metrics for **the human watching
  training**. Terminal output, live dashboard, resource tracking. It doesn't
  feed back into anything - it's purely observational.

| | Graph observation | Monitor |
|---|---|---|
| **Purpose** | Drive training decisions | Human-facing display |
| **Record** | `record()`/`collect()` per step, `flush()` per epoch | `log()` per epoch |
| **Analysis** | `trend().slope()`, `stalled()`, `improving()` | Raw history only |
| **Resources** | No | CPU, RAM, GPU, VRAM |
| **HTML output** | `plot_html()` - static chart of epoch curves | `save_html()` - full dashboard archive with resource graphs, epoch log, and graph SVG |
| **Live dashboard** | No | Yes (`serve()` with SSE streaming) |

They complement each other: use graph observation for metrics that drive
training decisions, and the monitor for human-facing output and system health.

### Using both together

`log` accepts a graph reference directly - it reads the latest epoch
history and forwards it to the monitor. You still flush yourself, so
observation and monitoring stay decoupled:

```rust
let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;
monitor.watch(&model);

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();

    for (input, target) in &batches {
        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;

        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;

        model.record_scalar("loss", loss.item()?);
    }

    // Observation: flush batch buffer into epoch history
    model.flush(&[]);

    // Training decisions use trends as usual
    if model.trend("loss").stalled(10, 1e-4) {
        optimizer.set_lr(scheduler.lr(epoch));
    }

    // Monitor: graph metrics + extras in one call
    monitor.log(epoch, t.elapsed(), (&model, &[("lr", scheduler.lr(epoch))]));
}

monitor.finish_with(&model);  // final SVG with profiling heat map
```

`log` accepts several forms via the [`Metrics`] trait:

```rust
// Plain metrics - no graph:
monitor.log(epoch, t.elapsed(), &[("loss", val), ("lr", lr)]);

// Graph only - all recorded metrics:
monitor.log(epoch, t.elapsed(), &model);

// Graph + extras - recorded metrics plus additional values:
monitor.log(epoch, t.elapsed(), (&model, &[("lr", lr)]));
```

## Complete Example

See [`flodl/examples/quickstart/`](../../flodl/examples/quickstart/) for
a runnable example with the monitor.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Utilities](08-utilities.md) | Next: [Graph Tree](10-graph-tree.md)
