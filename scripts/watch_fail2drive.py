"""Read archived official display frames through an SSH-forwardable browser stream.

No CARLA import, sensor, world tick, model call, or experiment mutation. This
script runs with the existing ep-fail2drive Python using only its standard library.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fail2Drive 画面转发</title><style>
body{margin:0;background:#101826;color:#e8eef7;font:16px system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:20px}h1{font-size:22px;margin:0 0 10px}
.note{color:#aebdd0;font-size:14px;line-height:1.6}.bar{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.tag{background:#24344d;padding:8px 12px;border-radius:6px}#status{background:#374253}
#screen{width:100%;min-height:200px;object-fit:contain;background:#070b12;border-radius:8px}
code{overflow-wrap:anywhere}#detail{line-height:1.7;font-size:14px;color:#c8d5e6;white-space:pre-wrap}
button{background:#355b87;color:white;border:0;padding:9px 12px;border-radius:5px;cursor:pointer}
</style><main><h1>Fail2Drive 官方展示相机</h1>
<div class="note">只读转发实验保存的展示画面，不推进仿真。相机归档频率为仿真时间2Hz；同步等待模型或规划时画面可以保持不动，等待时长不代表实验异常。下方身份对应服务端最新归档，浏览器画面传输可能稍有滞后。</div>
<div class="bar"><span class="tag" id="status">正在连接转发服务</span><span class="tag" id="age">尚无画面</span><button id="reconnect">重连画面</button></div>
<img id="screen" alt="等待实验产生展示相机画面"><p id="detail"></p></main><script>
const screen=document.getElementById('screen');let streamToken=null;
function connect(){screen.src='/stream?connection='+Date.now();}
document.getElementById('reconnect').onclick=connect;connect();
async function refresh(){try{const response=await fetch('/state',{cache:'no-store'});
 const s=await response.json();document.getElementById('status').textContent=s.message;
 document.getElementById('age').textContent=s.image_age_s==null?'尚无画面':'最新文件距今 '+s.image_age_s.toFixed(1)+' 秒';
 const fields=['路线：'+(s.route_run||'等待路线'),
 '最新归档的实际CARLA帧：'+(s.carla_frame??'未提供')+'；仿真时间：'+(s.carla_timestamp==null?'未提供':s.carla_timestamp.toFixed(3)+' 秒'),
 '相对物理步：'+(s.relative_tick??'未提供')+'；归档采样：2 Hz（仿真时间）',
 '来源：'+(s.image_path||'等待新帧')];if(s.warning)fields.push('说明：'+s.warning);
 document.getElementById('detail').textContent=fields.join('\n');
 if(streamToken!==s.route_run){streamToken=s.route_run;screen.removeAttribute('src');screen.hidden=true;connect();}
 screen.hidden=s.carla_frame==null;
 }catch(e){document.getElementById('status').textContent='转发连接中断，请检查SSH隧道或服务';}
 setTimeout(refresh,1000);}refresh();
</script></html>'''


class DisplayFeed:
    """A single cached poller for every browser; never reads outside the selected root."""
    def __init__(self, root, poll_s=.5, waiting_after_s=3.):
        self.root = Path(root).expanduser().resolve()
        self.poll_s, self.waiting_after_s = float(poll_s), float(waiting_after_s)
        self.lock = threading.Condition()
        self.stopping = threading.Event()
        self.image, self.image_meta, self.warning = None, None, None
        self.selected_manifest = None
        self.route_run = None
        self.route_folder = None
        self.last_scan = 0.
        self.thread = None

    def inside(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('artifact path escapes the selected results root')
        return path

    def _read_json(self, path):
        path = self.inside(path)
        if path.stat().st_size > 65536:
            raise ValueError('oversized display metadata')
        return json.loads(path.read_text(encoding='utf-8'))

    def discover_route(self):
        manifests = []
        # Avoid scanning thousands of camera/capture files and old viz images.
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('viz', 'captures', 'camera_frames',
                       '.git', '__pycache__', 'recorder', 'official_visualization')
                       and not (Path(directory)/d).is_symlink()]
            if 'official_visualization.json' in files:
                path = self.inside(Path(directory)/'official_visualization.json')
                manifests.append((path.stat().st_mtime_ns, str(path), path))
        if not manifests:
            return None, None
        path = max(manifests)[2]
        manifest = self._read_json(path)
        if manifest.get('source') != 'official_CARLA_rendered_camera':
            raise ValueError('display manifest has an unsupported source')
        folder = self.inside(manifest['path'])
        return path, folder

    def poll_once(self, force_scan=False):
        now = time.monotonic()
        try:
            if force_scan or now-self.last_scan >= 2.:
                manifest, folder = self.discover_route()
                self.last_scan = now
                if manifest != self.selected_manifest:
                    with self.lock:
                        self.selected_manifest, self.route_folder = manifest, folder
                        self.route_run = folder.name if folder is not None else None
                        self.image = self.image_meta = None
                        self.warning = None
                        self.lock.notify_all()
            if self.route_folder is None:
                return
            folder = self.inside(self.route_folder)
            # Names are relative physical ticks. True CARLA identity is mandatory
            # in the matching sidecar, never inferred from those names.
            candidates = [p for p in folder.glob('*.json') if re.fullmatch(r'\d{5,}\.json', p.name)]
            if not candidates:
                return
            path = max(candidates, key=lambda p:int(p.stem))
            path = self.inside(path)
            key = (str(path), path.stat().st_mtime_ns)
            with self.lock:
                if self.image_meta is not None and self.image_meta['_key'] == key:
                    return
            metadata = self._read_json(path)
            if metadata.get('schema') != 'ep-driver-f2d-display-frame-v1':
                raise ValueError('unsupported display-frame metadata schema')
            relative_tick, frame = metadata.get('relative_tick'), metadata.get('carla_frame')
            timestamp, saved = metadata.get('carla_timestamp'), metadata.get('saved_wall_time')
            if (type(relative_tick) is not int or relative_tick < 0 or relative_tick != int(path.stem)
                    or type(frame) is not int or frame < 0
                    or type(timestamp) not in (int,float) or not math.isfinite(timestamp)
                    or type(saved) not in (int,float) or not math.isfinite(saved)):
                raise ValueError('display source identity is incomplete or invalid')
            if (metadata.get('source') != 'official_CARLA_rendered_camera'
                    or metadata.get('camera_id') != 'rgb' or metadata.get('route_run') != folder.name
                    or metadata.get('sampling_hz_sim') != 2.):
                raise ValueError('display source/camera/route mismatch')
            jpeg_name = metadata.get('jpeg')
            if jpeg_name != path.with_suffix('.jpg').name:
                raise ValueError('display JPEG does not match its source metadata')
            image_path = self.inside(folder/jpeg_name)
            if image_path.stat().st_size > 16*1024*1024:
                raise ValueError('oversized display JPEG')
            image = image_path.read_bytes()
            if (not image.startswith(b'\xff\xd8') or not image.endswith(b'\xff\xd9')
                    or hashlib.sha256(image).hexdigest() != metadata.get('jpeg_sha256')):
                raise ValueError('display JPEG is incomplete or does not match its metadata')
            metadata['_key'] = key
            metadata['image_path'] = image_path.relative_to(self.root).as_posix()
            with self.lock:
                self.image, self.image_meta, self.warning = image, metadata, None
                self.lock.notify_all()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            with self.lock:
                self.warning = str(exc)
                self.lock.notify_all()

    def state(self):
        with self.lock:
            meta = self.image_meta
            if meta is None:
                return dict(status='waiting',message='等待本路线的新展示帧' if self.route_run else '等待实验创建路线',
                    route_run=self.route_run,image_age_s=None,warning=self.warning,
                    source_sampling_hz_sim=2.,carla_connected=False)
            age = max(0., time.time()-meta['saved_wall_time'])
            waiting = age >= self.waiting_after_s
            return dict(status='waiting_for_new_frame' if waiting else 'recent_frame',
                message='等待新帧（同步等待或实验已暂停／结束）' if waiting else '收到新的归档展示帧',
                route_run=self.route_run,image_age_s=age,carla_frame=meta['carla_frame'],
                carla_timestamp=meta['carla_timestamp'],relative_tick=meta['relative_tick'],
                image_path=meta['image_path'],warning=self.warning,source_sampling_hz_sim=2.,carla_connected=False)

    def start(self):
        self.poll_once(True)
        def loop():
            while not self.stopping.wait(self.poll_s):
                self.poll_once()
        self.thread = threading.Thread(target=loop, name='display-artifact-reader', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        with self.lock:self.lock.notify_all()
        if self.thread is not None:self.thread.join(timeout=5)


class DisplayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    def __init__(self, address, feed):
        self.feed = feed
        super().__init__(address, DisplayHandler)


class DisplayHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, fmt, *args):
        # Status is polled every second; avoid a growing per-frame access log.
        if args and any(str(args[0]).startswith('GET '+path) for path in ('/stream','/state')):return
        super().log_message(fmt,*args)

    def send(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(content)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == '/':
                self.send(PAGE.encode('utf-8'),'text/html; charset=utf-8')
            elif path == '/state':
                self.send(json.dumps(self.server.feed.state(),ensure_ascii=False,allow_nan=False).encode('utf-8'),
                          'application/json; charset=utf-8')
            elif path == '/stream':
                self.stream()
            else:
                self.send(b'Not found','text/plain; charset=utf-8',404)
        except (BrokenPipeError,ConnectionResetError,TimeoutError):
            pass

    def stream(self):
        self.connection.settimeout(10)
        self.send_response(200)
        self.send_header('Content-Type','multipart/x-mixed-replace; boundary=ep-frame')
        self.send_header('Cache-Control','no-store, no-cache, must-revalidate')
        self.send_header('Connection','close')
        self.end_headers()
        previous = None
        feed = self.server.feed
        while not feed.stopping.is_set():
            with feed.lock:
                meta = feed.image_meta
                if meta is None or meta['_key'] == previous:
                    feed.lock.wait(timeout=1)
                    continue
                image, previous = feed.image, meta['_key']
                # Never repeat old JPEGs to manufacture a playback frame rate.
                header = ('--ep-frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n'
                          'X-CARLA-Frame: %d\r\nX-CARLA-Time: %.6f\r\n\r\n' %
                          (len(image),meta['carla_frame'],meta['carla_timestamp'])).encode('ascii')
            self.wfile.write(header+image+b'\r\n');self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,help='Read-only method results root, including agent_runs and viz')
    parser.add_argument('--port',type=int,default=39080)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    feed = DisplayFeed(args.root)
    server = DisplayServer(('127.0.0.1',args.port),feed)
    feed.start()
    print('Fail2Drive只读画面转发：http://127.0.0.1:%d，结果根：%s' % (args.port,feed.root),flush=True)
    try:server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:pass
    finally:
        feed.stop();server.server_close()


if __name__ == '__main__':main()
