"""Local, offline video gallery. It never loads a scene or computes a score."""
import html
import json
from pathlib import Path

from .protocol import write_json


def update_gallery(output_dir):
    root = Path(output_dir).resolve()
    folder = root / "visualizations"
    folder.mkdir(parents=True, exist_ok=True)
    rows, complete, failed = [], 0, 0
    for replay in sorted((root / "samples").glob("*/official_replay")):
        status_file = replay / "render_status.json"
        try:
            status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {}
        except (OSError, ValueError):
            status = {}
        movie = replay / "official_bev.mp4"
        ok = status.get("status") == "complete" and movie.is_file() and movie.stat().st_size > 0
        complete += int(ok)
        failed += int(not ok)
        token = str(status.get("token", replay.parent.name))
        url = "../" + movie.relative_to(root).as_posix()
        trace = "../" + (replay / "trace.json.gz").relative_to(root).as_posix()
        label = str(status.get("stage", ""))
        control = ('<button class="play" data-src="%s" data-title="%s">Play BEV</button>' %
                   (html.escape(url, quote=True), html.escape(token + " / " + label, quote=True))) if ok else "Unavailable"
        rows.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td><a href="%s">Trace</a></td></tr>' %
                    (html.escape(token), html.escape(label), "complete" if ok else "failed / pending",
                     control, html.escape(trace, quote=True)))
    summary = dict(complete_videos=complete, failed_or_pending_videos=failed, total_replays=len(rows),
                   scoring_results_modified=False,
                   note="Only saved official simulation outputs; independent 4s anchors, not a continuous drive")
    write_json(folder / "replay_summary.json", summary)
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAVSIM official scoring BEV</title>
<style>body{font:15px system-ui,sans-serif;background:#eef2f6;color:#1d3045;margin:24px auto;max-width:1100px;padding:0 18px}
h1{font-size:25px}p{line-height:1.6}.panel{background:white;padding:18px;border-radius:12px;margin-bottom:18px}
video{display:block;width:min(100%,850px);max-height:75vh;margin:auto;background:#182637}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px 6px;text-align:left;border-bottom:1px solid #dce3eb}
button{background:#176f8c;color:white;border:0;border-radius:5px;padding:8px 12px;cursor:pointer}a{color:#176f8c}
input{padding:8px;width:min(440px,90%)}code{background:#edf1f5}</style>
<h1>NAVSIM official scoring: BEV execution replay</h1>
<p>Privileged-input baseline. Each video is an independent 0–4 s evaluation anchor.
It shows the official simulated ego and the traffic returned by the official policy;
it is not CARLA rendering or a continuous replanning drive.</p>
<div class="panel"><p>%(complete)d verified videos; %(failed)d failed/pending exports.
Visualization failures do not change official scores. Reload this page after more samples finish.
For a running evaluation, regenerate the gallery with <code>run_navsim.py gallery --output-dir ...</code>.</p>
<h2 id="selection">Select a sample below</h2><video id="player" controls preload="none"></video>
<p>Dashed plan and growing execution are different trajectories. Recorded human is reference only.
Raw diagnostic events precede human-penalty filtering; official final components apply to the whole anchor,
not the current frame. Two-stage aggregate EPDMS is available in the official result CSV.</p></div>
<div class="panel"><input id="filter" placeholder="Filter token or stage"><table>
<thead><tr><th>Token</th><th>Stage</th><th>Video status</th><th>Animation</th><th>Scoring trace</th></tr></thead>
<tbody id="rows">%(rows)s</tbody></table></div>
<script>document.querySelectorAll('.play').forEach(b=>b.addEventListener('click',()=>{
const p=document.getElementById('player');p.src=b.dataset.src;p.load();p.play().catch(()=>{});
document.getElementById('selection').textContent=b.dataset.title;p.scrollIntoView({behavior:'smooth',block:'center'});}));
document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();
document.querySelectorAll('#rows tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>"""
    document = (document.replace("%(complete)d", str(complete)).replace("%(failed)d", str(failed))
                .replace("%(rows)s", "\n".join(rows)))
    target = folder / "index.html"
    temporary = folder / "index.html.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(target)
    return str(target)
