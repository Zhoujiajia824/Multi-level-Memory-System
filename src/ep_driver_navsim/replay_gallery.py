"""Local, offline video gallery. It never loads a scene or computes a score."""
import html
import json
from pathlib import Path

from .protocol import write_json
from .replay_render import RENDERER_VERSION


def update_gallery(output_dir):
    root = Path(output_dir).resolve()
    folder = root / "visualizations"
    folder.mkdir(parents=True, exist_ok=True)
    rows, complete, failed, legacy = [], 0, 0, 0
    for replay in sorted((root / "samples").glob("*/official_replay")):
        status_file = replay / "render_status.json"
        try:
            status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {}
        except (OSError, ValueError):
            status = {}
        movie = replay / "official_bev.mp4"
        ok = status.get("status") == "complete" and movie.is_file() and movie.stat().st_size > 0
        official = ok and status.get("renderer_version") == RENDERER_VERSION and status.get("official_visualization") is True
        complete += int(official)
        legacy += int(ok and not official)
        failed += int(not ok)
        token = str(status.get("token", replay.parent.name))
        url = "../" + movie.relative_to(root).as_posix()
        preview = "../" + (replay / "official_bev_first.png").relative_to(root).as_posix()
        trace = "../" + (replay / "trace.json.gz").relative_to(root).as_posix()
        label = str(status.get("stage", ""))
        panel_path = replay / "official_cameras_anchor.png"
        camera_panel = "../" + panel_path.relative_to(root).as_posix() if official and panel_path.is_file() else ""
        cameras = [{"view": path.stem, "url": "../" + path.relative_to(root).as_posix()}
                   for path in sorted((replay.parent / "cameras").glob("*.jpg"))]
        control = ('<button class="play" data-official="%s" data-camera-panel="%s" data-src="%s" data-title="%s" data-poster="%s" data-cameras="%s">播放 BEV</button>' %
                   (str(bool(official)).lower(), html.escape(camera_panel, quote=True),
                    html.escape(url, quote=True), html.escape(token + " / " + label, quote=True),
                    html.escape(preview, quote=True), html.escape(json.dumps(cameras), quote=True))) if ok else "Unavailable"
        rows.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><a href="%s">Trace</a></td></tr>' %
                    (html.escape(token), html.escape(label), "official SDK plots" if official else ("legacy renderer; re-export for SDK style" if ok else "failed / pending"),
                     control, html.escape(trace, quote=True)))
    summary = dict(complete_videos=complete, legacy_style_videos=legacy, renderer_version=RENDERER_VERSION,
                   failed_or_pending_videos=failed, total_replays=len(rows),
                   scoring_results_modified=False,
                   note="Only saved official simulation outputs; independent 4s anchors, not a continuous drive")
    write_json(folder / "replay_summary.json", summary)
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAVSIM official scoring BEV</title>
<style>body{font:15px system-ui,sans-serif;background:#eef2f6;color:#1d3045;margin:24px auto;max-width:1100px;padding:0 18px}
h1{font-size:25px}p{line-height:1.6}.panel{background:white;padding:18px;border-radius:12px;margin-bottom:18px}
video{display:block;width:min(100%,850px);max-height:75vh;margin:auto;background:#182637}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px 6px;text-align:left;border-bottom:1px solid #dce3eb}
button{background:#176f8c;color:white;border:0;border-radius:5px;padding:8px 12px;cursor:pointer}a{color:#176f8c}
input{padding:8px;width:min(440px,90%)}code{background:#edf1f5}</style>
<style>#playback-error{color:#aa2538}#camera-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
figure{margin:0}figure img{width:100%;display:block}figcaption{padding:5px}summary{cursor:pointer}</style>
<h1>NAVSIM 官方评分：BEV 执行回放</h1>
<p>Privileged-input baseline. Each video is an independent 0–4 s evaluation anchor.
It shows the official simulated ego and the traffic returned by the official policy;
it is not CARLA rendering or a continuous replanning drive.</p>
<div class="panel"><p>%(complete)d verified official SDK visualizations; %(legacy)d legacy-style videos; %(failed)d failed/pending exports.
Visualization failures do not change official scores. Reload this page after more samples finish.
For a running evaluation, regenerate the gallery with <code>run_navsim.py gallery --output-dir ...</code>.</p>
<h2 id="selection">请选择样本</h2><video id="player" controls playsinline preload="metadata"></video>
<p id="playback-error" role="alert" aria-live="polite"></p>
<p id="renderer-label"></p>
<p><a id="direct-video" target="_blank" rel="noopener">单独打开视频</a> · <a id="download-video" download>下载视频</a> ·
<a id="preview-link" target="_blank" rel="noopener">打开 BEV 首帧</a></p>
<p>八路源锚点真实相机（原始/官方合成图）；这些静态图片对应 t=0，不随评分执行生成未来视角。</p>
<img id="camera-panel" alt="Eight static source-anchor cameras with official BEV" style="display:none;width:100%">
<details><summary>查看单独相机图片</summary><div id="camera-grid"></div></details>
<p>Official style: red dashed submitted plan, blue growing executed history, green recorded human reference.
Raw diagnostic events precede human-penalty filtering; official final components apply to the whole anchor,
not the current frame. Two-stage aggregate EPDMS is available in the official result CSV.</p></div>
<div class="panel"><input id="filter" placeholder="Filter token or stage"><table>
<thead><tr><th>Token</th><th>Stage</th><th>Video status</th><th>Animation</th><th>Scoring trace</th></tr></thead>
<tbody id="rows">%(rows)s</tbody></table></div>
<script>const player=document.getElementById('player'), error=document.getElementById('playback-error');
player.addEventListener('error',()=>{const codes={1:'播放已中止',2:'无法读取本地视频',3:'视频解码失败',4:'浏览器不支持视频编码或本地媒体访问'};
error.textContent=(codes[player.error?.code]||'视频播放失败')+'。可单独打开或下载视频；BEV首帧和相机图片仍可查看。';});
function select(b,autoplay){error.textContent='';player.poster=b.dataset.poster;player.src=b.dataset.src;player.load();
document.getElementById('direct-video').href=b.dataset.src;document.getElementById('download-video').href=b.dataset.src;
document.getElementById('preview-link').href=b.dataset.poster;
document.getElementById('selection').textContent=b.dataset.title;
document.getElementById('renderer-label').textContent=b.dataset.official==='true'?'NAVSIM SDK 官方绘图函数；实际评分轨迹与交通响应。':'旧版自定义配色；实际评分轨迹仍来自 trace，可离线重新导出官方样式。';
const panel=document.getElementById('camera-panel');panel.style.display=b.dataset.cameraPanel?'block':'none';
if(b.dataset.cameraPanel)panel.src=b.dataset.cameraPanel;else panel.removeAttribute('src');
const grid=document.getElementById('camera-grid');grid.replaceChildren();
JSON.parse(b.dataset.cameras).forEach(camera=>{const f=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption');
im.src=camera.url;im.alt=camera.view;im.loading='lazy';cap.textContent=camera.view;f.append(im,cap);grid.append(f);});
if(autoplay){player.play().catch(e=>{error.textContent='未能开始播放：'+e.message+'。请尝试播放器按钮或单独打开视频。';});
player.scrollIntoView({behavior:'smooth',block:'center'});}}
document.querySelectorAll('.play').forEach(b=>b.addEventListener('click',()=>select(b,true)));
const first=document.querySelector('.play[data-official="true"]')||document.querySelector('.play');if(first)select(first,false);
document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();
document.querySelectorAll('#rows tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>"""
    document = (document.replace("%(complete)d", str(complete)).replace("%(legacy)d", str(legacy)).replace("%(failed)d", str(failed))
                .replace("%(rows)s", "\n".join(rows)))
    target = folder / "index.html"
    temporary = folder / "index.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(target)
    return str(target)
