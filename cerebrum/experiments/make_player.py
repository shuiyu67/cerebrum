"""Turn a snake_run recording into a self-contained HTML player.

The recording stores, for every step, the snake's body and the *indices of
every neuron that fired*. Indices, not a bitmap: at fly scale 139k neurons a
frame would be 139 KB and 99.9% of it would be zeros.

The player draws two views side by side — the board the snake is playing on,
and every one of its neurons as a single pixel, decaying like a phosphor — so
the two can be read against each other. Watching the brain while the snake dies
is the point; a score alone would hide whether anything is happening inside.
"""

from __future__ import annotations

import argparse
import json
import os

TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>大脑玩贪吃蛇</title>
<style>
  :root {
    --bg: #14161a; --panel: #1b1e24; --line: #2c313a;
    --ink: #e8eaed; --dim: #8b919c; --accent: #3ddc97; --warn: #e05561;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 13px; line-height: 1.6;
  }
  .wrap { max-width: 780px; margin: 0 auto; padding: 20px 16px 40px; }
  h1 { font-size: 15px; font-weight: 500; margin: 0 0 4px; }
  .sub { color: var(--dim); font-size: 12px; margin-bottom: 16px; }
  .stage { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
  .col { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
  .label { color: var(--dim); font-size: 11px; margin-bottom: 6px; display: flex; justify-content: space-between; }
  canvas { display: block; border-radius: 6px; }
  #board { background: #101216; width: 280px; height: 280px; }
  #brain { background: #0d0f12; width: 373px; height: 373px; image-rendering: pixelated; }
  .controls { display: flex; gap: 8px; align-items: center; margin: 14px 0 10px; flex-wrap: wrap; }
  button {
    background: #252a32; color: var(--ink); border: 1px solid var(--line);
    border-radius: 6px; padding: 5px 12px; font-size: 12px; cursor: pointer;
    font-family: inherit;
  }
  button:hover { background: #2f3540; }
  button.primary { background: #1f5f47; border-color: #2b7f60; }
  input[type=range] { flex: 1; min-width: 160px; accent-color: var(--accent); }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 12px; }
  .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .stat .k { color: var(--dim); font-size: 11px; }
  .stat .v { font-size: 16px; font-variant-numeric: tabular-nums; }
  .v.good { color: var(--accent); }
  .v.bad { color: var(--warn); }
  .note { color: var(--dim); font-size: 11px; margin-top: 14px; line-height: 1.7; }
  code { background: #22262d; padding: 1px 5px; border-radius: 4px; font-size: 11px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>大脑玩贪吃蛇</h1>
  <div class="sub">__HEADLINE__</div>

  <div class="stage">
    <div class="col">
      <div class="label"><span>棋盘</span><span id="epLabel">—</span></div>
      <canvas id="board" width="280" height="280"></canvas>
    </div>
    <div class="col">
      <div class="label"><span>全部 __N__ 个神经元</span><span id="spikeLabel">—</span></div>
      <canvas id="brain" width="373" height="373"></canvas>
    </div>
  </div>

  <div class="controls">
    <button id="play" class="primary">播放</button>
    <button id="reset">重来</button>
    <input type="range" id="seek" min="0" max="__LAST__" value="0">
    <span id="posLabel" style="color:var(--dim);font-variant-numeric:tabular-nums">0 / __LAST__</span>
  </div>

  <div class="stats">
    <div class="stat"><div class="k">本局吃到的食物</div><div class="v" id="sAte">0</div></div>
    <div class="stat"><div class="k">动作（投票）</div><div class="v" id="sAct">—</div></div>
    <div class="stat"><div class="k">奖励</div><div class="v" id="sRew">0</div></div>
    <div class="stat"><div class="k">本步放电</div><div class="v" id="sSpk">0</div></div>
  </div>

  <div class="note">
    右边每个像素是一个真实的脉冲神经元，按动作分组上色：<span style="color:#e0556a">红 = 左转组</span> ·
    <span style="color:#3ddc97">绿 = 直行组</span> · <span style="color:#6fa8ff">蓝 = 右转组</span>。
    哪一块最亮，蛇就做哪个动作。亮起 = 这一步它放电了，然后按磷光衰减。
    每个时间步 = 1 毫秒生物时间，这个大脑里有 __SYN__ 条真实突触。
    左边棋盘上 <code>@</code> 是蛇头，颜色越亮越靠前。
  </div>
</div>

<script>
const DATA = __DATA__;
const F = DATA.frames;
const META = DATA.meta;
const N = META.neurons;
const GRID = 373;
const CELL = 20;

const board = document.getElementById('board');
const bctx = board.getContext('2d');
const brain = document.getElementById('brain');
const gctx = brain.getContext('2d', { alpha: false });
const gimg = gctx.createImageData(GRID, GRID);
const buf = new Float32Array(N);
const THIRD = Math.floor(N / 3);

const boardW = META.board_size * CELL;
board.width = boardW; board.height = boardW;
board.style.width = boardW + 'px'; board.style.height = boardW + 'px';

const seek = document.getElementById('seek');
const playBtn = document.getElementById('play');
seek.max = F.length - 1;

let idx = 0, playing = false, lastT = 0, speed = 3, acc = 0;

function decayAndSpike(frame) {
  for (let i = 0; i < N; i++) buf[i] *= 0.80;
  const sp = frame.sp;
  for (let k = 0; k < sp.length; k++) {
    const j = sp[k];
    if (j >= 0 && j < N) buf[j] = 1.0;
  }
  // Colour by actuator group. The brain is split into three equal blocks and
  // the block that fires most is the action, so tinting each block (left turn
  // / straight / right turn) makes the decision visible instead of implied.
  const d = gimg.data;
  for (let i = 0; i < N; i++) {
    const v = buf[i];
    const o = i << 2;
    const g = i < THIRD ? 0 : (i < 2 * THIRD ? 1 : 2);
    if (g === 0)      { d[o] = 14 + v * 200; d[o+1] = 16 + v * 90;  d[o+2] = 20 + v * 80; }
    else if (g === 1) { d[o] = 14 + v * 60;  d[o+1] = 16 + v * 230; d[o+2] = 20 + v * 150; }
    else              { d[o] = 14 + v * 70;  d[o+1] = 16 + v * 140; d[o+2] = 20 + v * 240; }
    d[o + 3] = 255;
  }
  gctx.putImageData(gimg, 0, 0);
}

function drawBoard(frame) {
  const S = META.board_size;
  bctx.fillStyle = '#101216';
  bctx.fillRect(0, 0, board.width, board.height);
  bctx.strokeStyle = '#1e222a';
  bctx.lineWidth = 1;
  for (let i = 1; i < S; i++) {
    bctx.beginPath();
    bctx.moveTo(i * CELL + 0.5, 0); bctx.lineTo(i * CELL + 0.5, board.height);
    bctx.moveTo(0, i * CELL + 0.5); bctx.lineTo(board.width, i * CELL + 0.5);
    bctx.stroke();
  }
  if (frame.food) {
    bctx.fillStyle = '#e05561';
    bctx.beginPath();
    bctx.arc(frame.food[0] * CELL + CELL / 2, frame.food[1] * CELL + CELL / 2,
             CELL * 0.28, 0, Math.PI * 2);
    bctx.fill();
  }
  const n = frame.x.length;
  for (let i = n - 1; i >= 0; i--) {
    const t = 1 - i / Math.max(1, n);
    bctx.fillStyle = i === 0 ? '#3ddc97'
                    : `rgba(61, 200, 140, ${0.25 + t * 0.55})`;
    bctx.fillRect(frame.x[i] * CELL + 2, frame.y[i] * CELL + 2, CELL - 4, CELL - 4);
  }
}

function drawStats(frame) {
  document.getElementById('epLabel').textContent = '第 ' + (frame.ep + 1) + ' 局';
  document.getElementById('spikeLabel').textContent = frame.n + ' 个放电';
  document.getElementById('sAte').textContent = frame.ate;
  const votes = frame.v || [0, 0, 0];
  const tot = votes[0] + votes[1] + votes[2] || 1;
  document.getElementById('sAct').textContent =
    ['左转', '直行', '右转'][frame.a] + '  ' +
    votes.map(v => Math.round(v / tot * 100) + '%').join(' / ');
  const r = frame.r;
  const el = document.getElementById('sRew');
  el.textContent = (r > 0 ? '+' : '') + r.toFixed(2);
  el.className = 'v ' + (r > 0 ? 'good' : r < -0.5 ? 'bad' : '');
  document.getElementById('sSpk').textContent = frame.n;
  document.getElementById('posLabel').textContent = idx + ' / ' + (F.length - 1);
}

function show(i) {
  idx = Math.max(0, Math.min(F.length - 1, i));
  const f = F[idx];
  drawBoard(f);
  decayAndSpike(f);
  drawStats(f);
  seek.value = idx;
}

function loop(t) {
  if (!playing) return;
  if (!lastT) lastT = t;
  const dt = t - lastT;
  lastT = t;
  acc += dt;
  const interval = 1000 / (speed * 12);
  while (acc >= interval && playing) {
    acc -= interval;
    if (idx >= F.length - 1) { playing = false; playBtn.textContent = '播放'; break; }
    show(idx + 1);
  }
  requestAnimationFrame(loop);
}

playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? '暂停' : '播放';
  lastT = 0; acc = 0;
  if (playing) {
    if (idx >= F.length - 1) show(0);
    requestAnimationFrame(loop);
  }
};
document.getElementById('reset').onclick = () => {
  playing = false; playBtn.textContent = '播放';
  buf.fill(0); show(0);
};
seek.oninput = () => {
  playing = false; playBtn.textContent = '播放';
  show(parseInt(seek.value, 10));
};
for (const k of [1, 2, 3, 5, 8]) { /* speed presets kept simple */ }
show(0);
</script>
</body>
</html>
"""


def build(record_path: str, out_path: str) -> dict:
    with open(record_path, encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload["meta"]
    frames = payload["frames"]

    # compact keys, so the file stays browser-sized
    slim = []
    for fr in frames:
        slim.append({
            "ep": fr["ep"],
            "x": fr["x"], "y": fr["y"], "food": fr["food"],
            "a": fr["a"], "r": fr["r"], "n": fr["n"],
            "ate": fr.get("ate", 0),
            "sp": fr["sp"],
        })

    headline = (
        f"果蝇级规模：{meta['neurons']:,} 个神经元 · {meta['synapses']:,} 条突触 · "
        f"{'学习开启' if meta['learn'] else '不学习（对照）'} · "
        f"{meta['episodes']} 局 · 平均吃到 {meta['score_mean']:.2f} 个"
    )

    html = (TEMPLATE
            .replace("__HEADLINE__", headline)
            .replace("__DATA__", json.dumps({"meta": meta, "frames": slim},
                                            separators=(",", ":"),
                                            ensure_ascii=False))
            .replace("__N__", f"{meta['neurons']:,}")
            .replace("__SYN__", f"{meta['synapses']:,}")
            .replace("__LAST__", str(max(0, len(slim) - 1))))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return {"html": out_path, "mb": round(os.path.getsize(out_path) / 1024 ** 2, 2),
            "frames": len(slim)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(build(a.record, a.out), ensure_ascii=False, indent=2))
