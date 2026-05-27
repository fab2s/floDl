# Tutorial 9: Training Monitor

The training monitor provides human-readable progress output, system resource
tracking, and an optional live web dashboard — all with zero external
dependencies.

> **Prerequisites**: [Training](04-training.md) covers the training loop.
> [Utilities](08-utilities.md) covers observation and trends.

## Basic Usage

The monitor wraps your training loop with timing, ETA, and resource sampling.
Record metrics on the graph during training, then pass the graph to `log`:

```rust
use flodl::*;
use flodl::monitor::Monitor;

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

GPU metrics (VRAM usage and utilization) appear automatically when CUDA is
available. On CPU-only builds they are silently omitted.

## Multiple Metrics

Record everything on the graph — `flush` averages each tag independently:

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
- **Training metrics chart**: live-updating canvas chart of all logged metrics
- **Resource chart**: CPU%, GPU%, RAM%, VRAM% over time
- **Resource bars**: current values with percentage fill
- **Epoch log table**: all epochs, newest first
- **Graph SVG**: collapsible architecture diagram (if provided)

### How it works

The server uses raw TCP sockets and [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-Sent_Events) (SSE) — no HTTP framework, no WebSocket library, no JavaScript dependencies.

1. `monitor.serve(port)` spawns a background listener thread
2. Each browser connection gets its own handler thread
3. `GET /` serves the dashboard HTML (all JS/CSS inline, ~16 KB)
4. `GET /events` holds the connection open as an SSE stream
5. Each `monitor.log(...)` pushes a JSON event to all connected clients
6. The browser JS updates the charts and log in real time

### Late join

If you open the dashboard mid-training, it catches up instantly. The SSE
handler replays all past epoch events before switching to live streaming.

### Multi-GPU and cluster runs

`monitor.serve(port)` works the same way on single-host multi-GPU and
multi-host clusters: **one URL covers the whole run**. When the
launcher fans out to multiple rank processes (auto-promoted on 2+
GPUs, or via `fdl cluster <cmd>`), the dashboard automatically grows
per-rank tabs labeled by host and local-rank (`host: lr=N gr=M`), with
throughput curves, batch-share distribution, VRAM, and ElChe anchor
evolution. No extra wiring — just open `http://<launcher-host>:3000`
and follow the whole cluster.

### Embedding the graph

To show the graph architecture in the dashboard:

```rust
let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;
monitor.watch(&model);  // generates SVG, sends to dashboard
```

The SVG appears in a collapsible section at the bottom of the dashboard.

For a timing-annotated heat map (green/yellow/red by execution time), enable
profiling during training and use `finish_with`:

```rust
model.enable_profiling();

for epoch in 0..num_epochs {
    // ... training (profiling records timing on each forward pass) ...
}

monitor.finish_with(&model);  // final SVG with steady-state timing heat map
```

`finish_with` generates the profiled SVG at the end of training — when the
last forward pass timing is representative of steady-state performance. The
heat map is pushed to the live dashboard and baked into the HTML archive.

You can also update the SVG mid-training with `watch_profiled(&model)`.

Both methods require Graphviz (`dot`) to be installed. If `dot` is not
available, they silently fall back or do nothing.

## Resource Tracking

The monitor samples system resources on every `log` call:

| Metric | Source | When available |
|--------|--------|----------------|
| CPU % | `/proc/stat` (delta) | Linux |
| RAM used/total | `/proc/meminfo` | Linux |
| GPU utilization % | NVML (dynamic load) | NVIDIA GPU + driver |
| VRAM allocated / spill | CUDA caching allocator (`reserved_bytes`) | CUDA feature enabled |

Resources that aren't available are silently omitted from both the terminal
output and the dashboard.

### VRAM metrics

flodl exposes two levels of CUDA memory measurement:

| Function | What it measures | PyTorch equivalent |
|----------|-----------------|-------------------|
| `cuda_active_bytes()` | Bytes backing live tensors | `torch.cuda.memory_allocated()` |
| `cuda_allocated_bytes()` | Total allocator reservation (includes cached free blocks) | `torch.cuda.memory_reserved()` |

The monitor tracks `cuda_allocated_bytes` (reserved) because it detects
unified-memory spill — when reserved bytes exceed physical VRAM, the
allocator has spilled to host RAM.

For debugging, compare both: if `active` is small but `reserved` is large,
the allocator is holding freed blocks. Call `cuda_empty_cache()` to release them.

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

Save the full dashboard as a self-contained HTML file — all charts, resource
graphs, epoch log, and graph SVG baked in. Open it in any browser, no server.

```rust
monitor.save_html("training_report.html");  // set before training
// ... training loop ...
monitor.finish();  // writes the archive
```

The archive is written automatically when `finish()` is called. It's the same
dashboard you see live, but frozen at the final state with all data pre-loaded.

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
naturally in the legend -- you can solo-click `subscan.ce` to focus on it.

If child subgraphs flush on a different cadence, use `flush_local()` to manage
them independently. See [Independent flush cadences](10-graph-tree.md#independent-flush-cadences).

## Monitor vs. Graph Observation

floDl has two metric systems that serve different purposes:

- **Graph observation** (`record`/`flush`/`trend`) — metrics that **feed back
  into training**. Use trends to trigger early stopping, LR decay, or
  convergence checks. The graph owns this data and your training loop reads it.

- **Monitor** (`log`/`serve`/`save_html`) — metrics for **the human watching
  training**. Terminal output, live dashboard, resource tracking. It doesn't
  feed back into anything — it's purely observational.

| | Graph observation | Monitor |
|---|---|---|
| **Purpose** | Drive training decisions | Human-facing display |
| **Record** | `record()`/`collect()` per step, `flush()` per epoch | `log()` per epoch |
| **Analysis** | `trend().slope()`, `stalled()`, `improving()` | Raw history only |
| **Resources** | No | CPU, RAM, GPU, VRAM |
| **HTML output** | `plot_html()` — static chart of epoch curves | `save_html()` — full dashboard archive with resource graphs, epoch log, and graph SVG |
| **Live dashboard** | No | Yes (`serve()` with SSE streaming) |

They complement each other: use graph observation for metrics that drive
training decisions, and the monitor for human-facing output and system health.

### Using both together

`log` accepts a graph reference directly — it reads the latest epoch
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
// Plain metrics — no graph:
monitor.log(epoch, t.elapsed(), &[("loss", val), ("lr", lr)]);

// Graph only — all recorded metrics:
monitor.log(epoch, t.elapsed(), &model);

// Graph + extras — recorded metrics plus additional values:
monitor.log(epoch, t.elapsed(), (&model, &[("lr", lr)]));
```

## Complete Example

See [`flodl/examples/quickstart/`](../../flodl/examples/quickstart/) for
a runnable example with the monitor.

---

Previous: [Tutorial 8: Utilities](08-utilities.md) |
Next: [Tutorial 10: Graph Tree](10-graph-tree.md)
