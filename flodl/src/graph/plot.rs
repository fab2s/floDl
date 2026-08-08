use std::fmt::Write;
use std::fs;
use std::path::Path;

use super::Graph;

/// A named data series for plotting or export.
struct PlotSeries {
    name: String,
    values: Vec<f64>,
}

impl Graph {
    /// Generate a self-contained HTML file with training curves from epoch history.
    ///
    /// This plots data accumulated via [`record()`](Self::record) /
    /// [`flush()`](Self::flush) — the graph's observation system for metrics
    /// that feed back into training decisions.
    ///
    /// For a full dashboard with resource graphs, epoch log, and graph SVG,
    /// use [`Monitor::save_html()`](crate::monitor::Monitor::save_html) instead.
    ///
    /// Tag group names are expanded. If tags is empty, all epoch history is plotted.
    /// Uses inline Canvas JS — no external dependencies.
    pub fn plot_html(&self, path: &str, tags: &[&str]) -> std::io::Result<()> {
        let series = self.gather_series(tags);
        if series.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no epoch data to plot (call flush first)",
            ));
        }
        let json = series_to_json(&series);
        let html = PLOT_TEMPLATE.replace("/*DATA*/", &json);
        fs::write(Path::new(path), html)
    }

    /// Generate a self-contained HTML file with per-node timing trend curves.
    ///
    /// Requires prior `collect_timings`/`flush_timings` calls. Tag groups are expanded.
    /// If tags is empty, all timing history is plotted.
    pub fn plot_timings_html(&self, path: &str, tags: &[&str]) -> std::io::Result<()> {
        let series = self.gather_timing_series(tags);
        if series.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no timing data to plot (call flush_timings first)",
            ));
        }
        let json = series_to_json(&series);
        let html = PLOT_TEMPLATE
            .replace("/*DATA*/", &json)
            .replace("Training Curves", "Timing Trends");
        fs::write(Path::new(path), html)
    }

    /// Export epoch history to CSV (one column per tag, one row per epoch).
    ///
    /// Tag groups are expanded. If tags is empty, all epoch history is exported.
    pub fn export_trends(&self, path: &str, tags: &[&str]) -> std::io::Result<()> {
        let series = self.gather_series(tags);
        if series.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no epoch data to export (call flush first)",
            ));
        }
        write_csv(path, &series)
    }

    /// Export timing epoch history to CSV (one column per tag, one row per epoch).
    ///
    /// Tag groups are expanded. If tags is empty, all timing history is exported.
    pub fn export_timing_trends(&self, path: &str, tags: &[&str]) -> std::io::Result<()> {
        let series = self.gather_timing_series(tags);
        if series.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no timing data to export (call flush_timings first)",
            ));
        }
        write_csv(path, &series)
    }

    /// Write a human-readable training log with per-epoch metrics and ETA.
    /// `total_epochs` is used for ETA (0 to omit).
    pub fn write_log(&self, path: &str, total_epochs: usize, tags: &[&str]) -> std::io::Result<()> {
        let series = self.gather_series(tags);
        if series.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no epoch data to log (call flush first)",
            ));
        }

        let mut b = String::with_capacity(4096);
        let _ = writeln!(b, "# flodl training log");

        let max_len = series.iter().map(|s| s.values.len()).max().unwrap_or(0);

        for i in 0..max_len {
            let _ = write!(b, "epoch {:3}", i + 1);
            for s in &series {
                if i < s.values.len() {
                    let _ = write!(b, "  {}={}", s.name, format_metric(s.values[i]));
                }
            }

            // Per-epoch timing from flush times if available
            let flush_times = self.flush_times.borrow();
            if i < flush_times.len() {
                let dur = if i == 0 {
                    flush_times[0]
                } else {
                    flush_times[i] - flush_times[i - 1]
                };
                if dur > 0.0 {
                    let _ = write!(b, "  [{}", format_duration(dur));
                    if total_epochs > 0 && i + 1 < total_epochs {
                        // Recent-pace ETA (mean of the last ≤5 epoch
                        // durations), matching the live monitor: the old
                        // global `elapsed / (i+1)` average baked epoch 0's
                        // startup (data load, init, calibration) into every
                        // line, so the ETA started inflated and melted.
                        let lo = i.saturating_sub(4);
                        let mut acc = 0.0;
                        let mut n = 0usize;
                        for j in lo..=i {
                            let d = if j == 0 {
                                flush_times[0]
                            } else {
                                flush_times[j] - flush_times[j - 1]
                            };
                            if d > 0.0 {
                                acc += d;
                                n += 1;
                            }
                        }
                        if n > 0 {
                            let remaining = (acc / n as f64) * (total_epochs - i - 1) as f64;
                            if remaining > 0.0 {
                                let _ = write!(b, "  ETA {}", format_duration(remaining));
                            }
                        }
                    }
                    b.push(']');
                }
            }
            b.push('\n');
        }

        fs::write(Path::new(path), b)
    }

    // --- Internal helpers ---

    fn gather_series(&self, tags: &[&str]) -> Vec<PlotSeries> {
        let history = self.epoch_history.borrow();
        let expanded = if tags.is_empty() {
            let mut all: Vec<String> = history.keys().cloned().collect();
            all.sort();
            all
        } else {
            self.expand_groups(tags)
        };

        expanded
            .into_iter()
            .filter_map(|tag| {
                history.get(&tag).and_then(|vals| {
                    if vals.is_empty() {
                        None
                    } else {
                        Some(PlotSeries {
                            name: tag,
                            values: vals.clone(),
                        })
                    }
                })
            })
            .collect()
    }

    fn gather_timing_series(&self, tags: &[&str]) -> Vec<PlotSeries> {
        let history = self.timing_history.borrow();
        let expanded = if tags.is_empty() {
            let mut all: Vec<String> = history.keys().cloned().collect();
            all.sort();
            all
        } else {
            self.expand_groups(tags)
        };

        expanded
            .into_iter()
            .filter_map(|tag| {
                history.get(&tag).and_then(|vals| {
                    if vals.is_empty() {
                        None
                    } else {
                        Some(PlotSeries {
                            name: tag,
                            values: vals.clone(),
                        })
                    }
                })
            })
            .collect()
    }
}

/// Format duration: <1s → "42ms", <1m → "1.2s", ≥1m → "2m05s".
pub fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        format!("{}ms", (secs * 1000.0) as u64)
    } else if secs < 60.0 {
        format!("{:.1}s", secs)
    } else {
        let mins = (secs / 60.0) as u64;
        let rem = secs as u64 % 60;
        format!("{}m{:02}s", mins, rem)
    }
}

/// Adaptive float formatting for log display.
fn format_metric(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".to_string()
    } else if abs < 0.001 {
        format!("{:.2e}", v)
    } else if abs < 100.0 {
        format!("{:.4}", v)
    } else {
        format!("{:.2}", v)
    }
}

/// Serialize plot series to JSON (no serde dependency).
fn series_to_json(series: &[PlotSeries]) -> String {
    let mut b = String::from("[");
    for (i, s) in series.iter().enumerate() {
        if i > 0 {
            b.push(',');
        }
        let _ = write!(b, "{{\"name\":\"{}\",\"values\":[", s.name);
        for (j, v) in s.values.iter().enumerate() {
            if j > 0 {
                b.push(',');
            }
            let _ = write!(b, "{:.8}", v);
        }
        b.push_str("]}");
    }
    b.push(']');
    b
}

/// Write series data to CSV.
fn write_csv(path: &str, series: &[PlotSeries]) -> std::io::Result<()> {
    let mut b = String::with_capacity(4096);
    b.push_str("epoch");
    for s in series {
        b.push(',');
        b.push_str(&s.name);
    }
    b.push('\n');

    let max_len = series.iter().map(|s| s.values.len()).max().unwrap_or(0);
    for i in 0..max_len {
        let _ = write!(b, "{}", i + 1);
        for s in series {
            b.push(',');
            if i < s.values.len() {
                let _ = write!(b, "{:.8}", s.values[i]);
            }
        }
        b.push('\n');
    }

    fs::write(Path::new(path), b)
}

/// Self-contained HTML template for training curves.
/// `/*DATA*/` is replaced with JSON series data.
const PLOT_TEMPLATE: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Training Curves</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Helvetica,Arial,sans-serif;background:#f5f6fa;padding:24px}
.container{background:#fff;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.08);padding:24px;max-width:960px;margin:0 auto}
h2{color:#2c3e50;margin-bottom:16px;font-size:18px}
canvas{width:100%;cursor:crosshair}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:14px}
.legend-item{display:flex;align-items:center;gap:6px;font-size:12px;color:#555}
.legend-color{width:12px;height:12px;border-radius:2px}
.tooltip{position:absolute;background:rgba(44,62,80,.9);color:#fff;padding:6px 10px;border-radius:4px;font-size:11px;pointer-events:none;display:none;white-space:nowrap}
</style>
</head>
<body>
<div class="container">
<h2>Training Curves</h2>
<canvas id="chart" height="400"></canvas>
<div class="legend" id="legend"></div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA=/*DATA*/;
const COLORS=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#34495e','#c0392b','#2980b9','#27ae60','#d35400'];
const canvas=document.getElementById('chart');
const ctx=canvas.getContext('2d');
const tooltip=document.getElementById('tooltip');
const legend=document.getElementById('legend');
const dpr=window.devicePixelRatio||1;
function resize(){
  const rect=canvas.getBoundingClientRect();
  canvas.width=rect.width*dpr;
  canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);
  draw();
}
const M={top:20,right:20,bottom:36,left:60};
function draw(){
  const W=canvas.width/dpr,H=canvas.height/dpr;
  const pw=W-M.left-M.right,ph=H-M.top-M.bottom;
  ctx.clearRect(0,0,W,H);
  if(!DATA||DATA.length===0)return;
  let maxEp=0,minV=Infinity,maxV=-Infinity;
  DATA.forEach(s=>{
    maxEp=Math.max(maxEp,s.values.length);
    s.values.forEach(v=>{minV=Math.min(minV,v);maxV=Math.max(maxV,v)});
  });
  if(minV===maxV){minV-=1;maxV+=1}
  const pad=(maxV-minV)*0.05;
  minV-=pad;maxV+=pad;
  const xScale=i=>M.left+(i/(maxEp-1||1))*pw;
  const yScale=v=>M.top+ph-(v-minV)/(maxV-minV)*ph;
  ctx.strokeStyle='#eee';ctx.lineWidth=1;
  const yTicks=5;
  for(let i=0;i<=yTicks;i++){
    const v=minV+(maxV-minV)*i/yTicks;
    const y=yScale(v);
    ctx.beginPath();ctx.moveTo(M.left,y);ctx.lineTo(W-M.right,y);ctx.stroke();
    ctx.fillStyle='#999';ctx.font='10px Helvetica';ctx.textAlign='right';
    ctx.fillText(formatVal(v),M.left-6,y+3);
  }
  const xStep=Math.max(1,Math.floor(maxEp/10));
  ctx.textAlign='center';
  for(let i=0;i<maxEp;i+=xStep){
    const x=xScale(i);
    ctx.beginPath();ctx.moveTo(x,M.top);ctx.lineTo(x,H-M.bottom);ctx.stroke();
    ctx.fillStyle='#999';ctx.font='10px Helvetica';
    ctx.fillText(''+(i+1),x,H-M.bottom+14);
  }
  ctx.fillStyle='#888';ctx.font='11px Helvetica';ctx.textAlign='center';
  ctx.fillText('Epoch',M.left+pw/2,H-4);
  DATA.forEach((s,si)=>{
    const color=COLORS[si%COLORS.length];
    ctx.strokeStyle=color;ctx.lineWidth=2;
    ctx.beginPath();
    s.values.forEach((v,i)=>{
      const x=xScale(i),y=yScale(v);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
    s.values.forEach((v,i)=>{
      ctx.fillStyle=color;ctx.beginPath();
      ctx.arc(xScale(i),yScale(v),3,0,Math.PI*2);ctx.fill();
    });
  });
  canvas._layout={xScale,yScale,maxEp,minV,maxV,pw,ph};
}
function formatVal(v){
  if(Math.abs(v)<0.001&&v!==0)return v.toExponential(1);
  if(Math.abs(v)>=1000)return v.toFixed(0);
  if(Math.abs(v)>=1)return v.toFixed(3);
  return v.toFixed(4);
}
DATA.forEach((s,i)=>{
  const item=document.createElement('div');item.className='legend-item';
  const swatch=document.createElement('div');swatch.className='legend-color';
  swatch.style.background=COLORS[i%COLORS.length];
  const label=document.createTextNode(s.name);
  item.appendChild(swatch);item.appendChild(label);
  legend.appendChild(item);
});
canvas.addEventListener('mousemove',e=>{
  const L=canvas._layout;if(!L)return;
  const rect=canvas.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  let bestDist=Infinity,bestEp=-1;
  for(let i=0;i<L.maxEp;i++){
    const d=Math.abs(L.xScale(i)-mx);
    if(d<bestDist){bestDist=d;bestEp=i}
  }
  if(bestDist>20){tooltip.style.display='none';return}
  let html='<b>Epoch '+(bestEp+1)+'</b>';
  DATA.forEach((s,si)=>{
    if(bestEp<s.values.length){
      const c=COLORS[si%COLORS.length];
      html+='<br><span style="color:'+c+'">■</span> '+s.name+': '+formatVal(s.values[bestEp]);
    }
  });
  tooltip.innerHTML=html;tooltip.style.display='block';
  let tx=e.pageX+12,ty=e.pageY-10;
  const tw=tooltip.offsetWidth,th=tooltip.offsetHeight;
  if(tx+tw>window.innerWidth+window.scrollX)tx=e.pageX-tw-12;
  if(ty+th>window.innerHeight+window.scrollY)ty=e.pageY-th-10;
  tooltip.style.left=tx+'px';tooltip.style.top=ty+'px';
});
canvas.addEventListener('mouseleave',()=>{tooltip.style.display='none'});
window.addEventListener('resize',resize);
resize();
</script>
</body>
</html>"##;
