"""A bounded, display-only browser preview of already composed BGR frames.

No CARLA client, sensor or planner is accessed here. A single JPEG worker owns
encoding; HTTP clients consume immutable bytes and can never hold up publish.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np


_PAGE = b"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CARLA live trajectory view</title><style>
body{margin:0;background:#151c24;color:#dce4ed;font:16px system-ui}
main{max-width:1440px;margin:auto;padding:16px}h1{font-size:20px;margin:0 0 10px}
#state{padding:10px;background:#25313e;border-radius:6px;margin-bottom:12px}
img{display:block;width:100%;height:auto;background:#202a35;min-height:120px}
small{display:block;margin-top:10px;color:#aab8c9}</style><main>
<h1>CARLA live trajectory view</h1><div id="state">Connecting / waiting for a composed frame</div>
<img id="frame" alt="Waiting for a captured CARLA frame">
<small>Display camera only. Frame and simulation time above belong to this captured image.
Selected and accepted lines are future plans, not the vehicle's travelled path.</small>
</main><script>
let busy=false,last=null,objectURL=null;async function refresh(){if(busy)return;busy=true;
try{const r=await fetch('/frame.jpg',{cache:'no-store'});
if(r.status===503){document.getElementById('state').textContent='Connected | waiting for a composed frame';return;}
if(!r.ok)throw Error();const m=JSON.parse(r.headers.get('X-Frame-Metadata')||'{}');
const sequence=r.headers.get('X-Live-Sequence'),age=Number(r.headers.get('X-Frame-Age'));
if(sequence!==last){const blob=await r.blob(),next=URL.createObjectURL(blob),img=document.getElementById('frame');
await new Promise((ok,bad)=>{img.onload=ok;img.onerror=bad;img.src=next;});
if(objectURL)URL.revokeObjectURL(objectURL);objectURL=next;last=sequence;}
const f=m.carla_frame_id??m.frame_id??'-',t=typeof m.sim_time_s==='number'?m.sim_time_s.toFixed(2)+' s':'-';
document.getElementById('state').textContent='Connected | '+r.headers.get('X-Live-State')+
' | CARLA frame '+f+' | simulation '+t+' | frame age '+age.toFixed(1)+' s';
}catch(e){document.getElementById('state').textContent='Disconnected / preview server unavailable';}
finally{busy=false;}}refresh();setInterval(refresh,150);
</script></html>"""


class LiveTrajectoryView:
    """Serve composed video frames without coupling network IO to control.

    Config keys: enabled (default False), host (127.0.0.1), port (8765; 0
    requests an ephemeral port), jpeg_quality (85), stale_after_s (2.0).
    ``start`` returns the browser URL. ``publish`` copies both inputs and uses
    a one-frame queue, dropping superseded preview frames. It never waits for
    JPEG encoding or a client. Call ``close`` once the owning run ends.
    """

    def __init__(self, config=None):
        config = dict(config or {})
        self.enabled = config.get("enabled", False)
        if type(self.enabled) is not bool:
            raise ValueError("live view enabled must be boolean")
        self.host = str(config.get("host", "127.0.0.1"))
        self.port = int(config.get("port", 8765))
        self.jpeg_quality = int(config.get("jpeg_quality", 85))
        self.stale_after_s = float(config.get("stale_after_s", 2.0))
        if not self.host or not 0 <= self.port <= 65535:
            raise ValueError("live view requires a host and port in [0, 65535]")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("live view JPEG quality must be in [1, 100]")
        if not math.isfinite(self.stale_after_s) or self.stale_after_s <= 0:
            raise ValueError("live view stale interval must be positive and finite")
        self.url = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=1)
        self._server = self._server_thread = self._worker = None
        self._started = self._closed = False
        self._latest = None
        self._published = self._encoded = self._dropped = 0
        self._publish_errors = self._jpeg_errors = self._clients = 0
        self._last_error = None

    def start(self):
        if not self.enabled:
            return None
        if self._closed:
            raise RuntimeError("A closed live trajectory view cannot be restarted")
        if self._started:
            return self.url
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(2.0)

            def _send(self, status, mime, payload, headers=None):
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                try:
                    path = urlsplit(self.path).path
                    if path == "/":
                        self._send(200, "text/html; charset=utf-8", _PAGE)
                    elif path == "/status.json":
                        self._send(200, "application/json", json.dumps(owner.summary(),
                            ensure_ascii=False, allow_nan=False).encode("utf-8"))
                    elif path == "/frame.jpg":
                        with owner._condition:
                            latest = owner._latest
                        headers = None
                        if latest:
                            fields = ("carla_frame_id", "frame_id", "sim_time_s", "run_id",
                                      "planner_sequence", "selected_id")
                            metadata = {key: latest[2][key] for key in fields if key in latest[2]}
                            age = max(0., time.monotonic() - latest[3])
                            headers = {"X-Frame-Metadata": json.dumps(metadata, ensure_ascii=True),
                                       "X-Live-Sequence": str(latest[0]), "X-Frame-Age": str(age),
                                       "X-Live-State": "stale" if age > owner.stale_after_s else "live"}
                        self._send(200 if latest else 503, "image/jpeg" if latest else "application/json",
                                   latest[1] if latest else b'{"state":"waiting"}', headers)
                    elif path in ("/stream.mjpg", "/stream"):
                        self._stream()
                    else:
                        self._send(404, "text/plain; charset=utf-8", b"Not found")
                except (BrokenPipeError, ConnectionError, OSError):
                    pass  # A slow or disconnected browser is not a run error.

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                sequence = -1
                with owner._condition:
                    owner._clients += 1
                try:
                    while not owner._stop.is_set():
                        with owner._condition:
                            owner._condition.wait_for(lambda: owner._stop.is_set() or
                                (owner._latest is not None and owner._latest[0] != sequence), timeout=.5)
                            latest = owner._latest
                            if owner._stop.is_set():
                                break
                            if latest is None or latest[0] == sequence:
                                continue
                            sequence, jpeg, metadata, _, _ = latest
                        # Immutable bytes are written after releasing the lock.
                        fid = metadata.get("carla_frame_id", metadata.get("frame_id"))
                        header = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                  str(len(jpeg)).encode("ascii") + b"\r\n")
                        if isinstance(fid, int) and not isinstance(fid, bool):
                            header += b"X-Carla-Frame: " + str(fid).encode("ascii") + b"\r\n"
                        self.wfile.write(header + b"\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                finally:
                    with owner._condition:
                        owner._clients -= 1

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.daemon_threads = True
        server.block_on_close = False
        self._server = server
        self.port = int(server.server_address[1])
        url_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        self.url = f"http://{url_host}:{self.port}/"
        self._worker = threading.Thread(target=self._encode, name="carla-live-jpeg", daemon=True)
        self._server_thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.1),
                                               name="carla-live-http", daemon=True)
        try:
            self._worker.start()
            self._server_thread.start()
            self._started = True
        except BaseException:
            self.close()
            raise
        return self.url

    def publish(self, image, metadata=None):
        if not self.enabled or not self._started or self._closed:
            return False
        try:
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
                    or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
                raise ValueError("Live view input must be a nonempty uint8 BGR image")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("Live view metadata must be a dictionary")
            detached = json.loads(json.dumps(metadata or {}, ensure_ascii=False, allow_nan=False))
            pixels = image.copy()
            with self._condition:
                if self._closed:
                    return False
                try:
                    self._queue.put_nowait((pixels, detached))
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                        self._dropped += 1
                    except queue.Empty:
                        pass
                    self._queue.put_nowait((pixels, detached))
                self._published += 1
            return True
        except Exception as exc:
            with self._condition:
                self._publish_errors += 1
                self._last_error = type(exc).__name__
            return False

    def _encode(self):
        while not self._stop.is_set():
            try:
                pixels, metadata = self._queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                ok, encoded = cv2.imencode(".jpg", pixels,
                                         [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    raise RuntimeError("JPEG encoder rejected preview frame")
                jpeg = encoded.tobytes()
                with self._condition:
                    if self._stop.is_set():
                        break
                    self._encoded += 1
                    self._latest = (self._encoded, jpeg, metadata, time.monotonic(), time.time())
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._jpeg_errors += 1
                    self._last_error = type(exc).__name__

    def summary(self):
        with self._condition:
            latest = self._latest
            age = max(0., time.monotonic() - latest[3]) if latest else None
            state = ("disabled" if not self.enabled else "closed" if self._closed else
                     "waiting" if latest is None else "stale" if age > self.stale_after_s else "live")
            return {"enabled": self.enabled, "started": self._started, "closed": self._closed,
                    "url": self.url, "host": self.host, "port": self.port, "state": state,
                    "client_count": self._clients, "queue_capacity": 1,
                    "pending_frame_count": self._queue.qsize(), "published_frame_count": self._published,
                    "encoded_frame_count": self._encoded, "dropped_frame_count": self._dropped,
                    "publish_error_count": self._publish_errors, "jpeg_error_count": self._jpeg_errors,
                    "last_error": self._last_error, "last_frame_age_s": age,
                    "encoded_at_unix_s": latest[4] if latest else None,
                    "frame": json.loads(json.dumps(latest[2])) if latest else None}

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            self._condition.notify_all()
        if self._server is not None:
            if self._server_thread is not None and self._server_thread.is_alive():
                self._server.shutdown()
            self._server.server_close()
        for thread in (self._worker, self._server_thread):
            if thread is not None and thread is not threading.current_thread() and thread.is_alive():
                thread.join(timeout=2.)
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
