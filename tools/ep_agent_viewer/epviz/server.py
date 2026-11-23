"""Independent HTTP server. It never connects to or advances a simulator."""
from __future__ import annotations

import argparse
import copy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import socket
import threading
from urllib.parse import parse_qs, quote, unquote, urlsplit
import webbrowser

from . import __version__
from .episodes import EpisodeStore
from .media import MediaCache

WEB = Path(__file__).resolve().parents[1] / "web"
PRIVATE_KEYS = re.compile(r"(^|_)(api_?key|secret|password|authorization|access_token|refresh_token)($|_)", re.I)


def safe_json(value):
    """Keep arrays JSON-safe and exclude configuration credentials from the UI."""
    if isinstance(value, dict):
        return {str(k): "[redacted]" if PRIVATE_KEYS.search(str(k)) else safe_json(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return safe_json(value.tolist())
    return value


def library_wire_frame(frame):
    """Keep rendered geometry and provenance without sending duplicate source maps."""
    result = copy.deepcopy(frame)
    scene = result.get('scene')
    if isinstance(scene, dict):
        result['scene'] = {key: scene[key] for key in ('frame_id', 'time', 'scenario', 'coordinates') if key in scene}
    for actor in [result.get('ego'), *(result.get('actors') or [])]:
        if isinstance(actor, dict):
            actor.pop('history', None)
    visualization = result.get('visualization') or {}
    visualization['roads'] = []  # The exact merged render_geometry is already present.
    render = visualization.get('render_geometry') or {}
    if visualization.get('topology') == render.get('topology'):
        visualization.pop('topology', None)  # Renderer also reads render_geometry.topology.
    surface = render.get('road_surface') or {}
    if surface.get('recorded_polygons') == surface.get('polygons'):
        surface.pop('recorded_polygons', None)
    for marking in render.get('markings') or []:
        for key in ('source_boundary_ids', 'source_lane_ids'):
            values = marking.pop(key, None)
            if isinstance(values, list):
                marking[key + '_count'] = len(values)
    return result


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    # Windows SO_REUSEADDR permits a second listener to bind an occupied port.
    # Fail clearly instead of sharing a port with another local application.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, address, root=None, engine_root=None, cache_dir=None, ffmpeg=None, memory_root=None):
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.engine_root = Path(engine_root).expanduser().resolve() if engine_root else None
        self.store = EpisodeStore(self.root, engine_root=self.engine_root) if self.root else None
        self.store_lock = threading.RLock()
        self.library_lock = threading.RLock()
        self.library = None
        if memory_root is not None:
            from .library import LibraryStore
            self.library = LibraryStore(memory_root)
        self.media = MediaCache(cache_dir, ffmpeg)
        super().__init__(address, ViewerHandler)

    @staticmethod
    def asset_url(episode, relative):
        return "/api/asset?episode=" + quote(str(episode), safe="") + "&path=" + quote(str(relative), safe="")

    def decorate(self, episode, document):
        """Attach browser asset URLs without exposing arbitrary filesystem access."""
        result = copy.deepcopy(document)
        assets = result.get("assets")
        if isinstance(assets, dict):
            for key, value in list(assets.items()):
                if isinstance(value, str):
                    assets[key + "_url"] = self.asset_url(episode, value)
                elif isinstance(value, dict):
                    path = value.get("path") or value.get("relative_path")
                    if path:
                        value["url"] = self.asset_url(episode, path)
        cameras = result.get("cameras", [])
        iterable = cameras.values() if isinstance(cameras, dict) else cameras
        for camera in iterable:
            if not isinstance(camera, dict):
                continue
            path = camera.get("path") or camera.get("rgb_path") or camera.get("relative_path")
            if path:
                camera["url"] = self.asset_url(episode, path)
        video = result.get("video")
        if isinstance(video, dict) and video.get("preview_path"):
            video["preview_url"] = self.asset_url(episode, video["preview_path"])
        if isinstance(video, dict) and video.get("path"):
            video["url"] = self.asset_url(episode, video["path"])
            if video.get("revision"):
                video["url"] += "&revision=" + quote(str(video["revision"]), safe="")
            try:
                source = self.store.resolve_asset(episode, video["path"])
                video["compatibility"] = self.media.status(source)
                if video["compatibility"]["status"] == "ready":
                    video["original_url"] = video["url"]
                    video["url"] = video["compatibility"]["url"]
            except (OSError, ValueError):
                pass
        for field in ("model_input", "active_model_input"):
            value = result.get(field)
            if isinstance(value, dict):
                value["image_urls"] = [self.asset_url(episode, path) for path in value.get("image_paths", [])]
        result.setdefault("episode_id", episode)
        if "frames" in result and "times" not in result:
            result["times"] = [row.get("time") for row in result["frames"]]
        return result


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "EPAgentViewer/" + __version__

    def do_HEAD(self):
        self._dispatch(head=True)

    def do_GET(self):
        self._dispatch(head=False)

    def do_POST(self):
        self._dispatch(head=False, post=True)

    def _json(self, document, status=200, head=False):
        data = json.dumps(safe_json(document), ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _dispatch(self, head=False, post=False):
        try:
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query)
            if path.startswith('/api/library'):
                if post:
                    return self._json(dict(error="离线记忆库仅支持只读浏览"), 405, head)
                if self.server.library is None:
                    return self._json(dict(available=False, read_only=True,
                        reason="启动时通过 --memory-root 指定离线记忆库目录。"), head=head)
                with self.server.library_lock:
                    if path == '/api/library':
                        document = dict(self.server.library.summary())
                        document.setdefault('available', True)
                        document['experiments_configured'] = self.server.store is not None
                        return self._json(document, head=head)
                    if path == '/api/library/records':
                        document = self.server.library.records(query=query.get('q', [''])[0],
                            source=query.get('source', [''])[0], status=query.get('status', [''])[0],
                            offset=int(query.get('offset', ['0'])[0]), limit=int(query.get('limit', ['40'])[0]))
                        return self._json(document, head=head)
                    if path == '/api/library/asset':
                        asset = self.server.library.resolve_asset(query.get('id', [''])[0], query.get('path', [''])[0])
                        return self._file(asset, head)
                    scene_match = re.fullmatch(r'/api/library/records/([^/]+)/scene', path)
                    if scene_match:
                        document = self.server.library.scene_data(scene_match[1],
                            evidence_index=int(query.get('evidence', ['0'])[0]))
                        if document.get('available'):
                            from .library_scene import build_library_scene
                            scene, grid = document.pop('scene'), document.pop('grid', None)
                            future, camera_data = document.pop('future', None), document.pop('camera_data', None)
                            frame = build_library_scene(scene, grid=grid, future=future)
                            from .library_projection import attach_library_cameras
                            frame = library_wire_frame(attach_library_cameras(frame, camera_data))
                            document = {**document, **frame}
                        return self._json(document, head=head)
                    record_match = re.fullmatch(r'/api/library/records/([^/]+)', path)
                    if record_match:
                        document = self.server.library.detail(record_match[1],
                            evidence_index=int(query.get('evidence', ['0'])[0]))
                        return self._json(document, head=head)
                raise FileNotFoundError('未知的记忆库接口')
            if self.server.store is None and (path.startswith('/api/episodes/') or path == '/api/asset' or
                                               re.match(r'^/assets/[0-9a-f]{16}/', path)):
                raise FileNotFoundError('未指定实验目录；当前服务仅浏览离线记忆库')
            video_match = re.fullmatch(r"/api/episodes/([^/]+)/video", path)
            if video_match:
                with self.server.store_lock:
                    manifest = self.server.store.get_manifest(video_match[1])
                    video = manifest.get("video") or {}
                    if not video.get("available"):
                        return self._json(dict(status="unavailable", error="此轮尚无录像"), head=head)
                    if not manifest.get("summary") or video.get("status") not in (None, "complete"):
                        return self._json(dict(status="unavailable", error="录像仍在录制或尚未完整封装；结束后再生成副本"), head=head)
                    source = self.server.store.resolve_asset(video_match[1], video["path"])
                document = self.server.media.status(source, start=post)
                return self._json(document, 202 if document["status"] == "working" else 200, head=head)
            if post:
                return self._json(dict(error="此端点不支持写入"), 405, head)
            if path in ('/', '/index.html') and self.server.store is None and self.server.library is not None:
                self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
                self.send_header('Location', '/memory' + ('?' + parsed.query if parsed.query else ''))
                self.send_header('Content-Length', '0')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                return
            if path.startswith("/api/video-cache/"):
                return self._file(self.server.media.resolve(path.rsplit("/", 1)[-1]), head)
            if path == "/api/health":
                return self._json(dict(status="ok", version=__version__, read_only=True), head=head)
            if path == "/api/episodes":
                with self.server.store_lock:
                    episodes = self.server.store.discover() if self.server.store else []
                    runs = self.server.store.list_runs() if self.server.store else []
                return self._json(dict(episodes=episodes, runs=runs, read_only=True), head=head)
            run_match = re.fullmatch(r"/api/runs/([0-9a-f]{16})", path)
            if run_match:
                if self.server.store is None:
                    raise FileNotFoundError("未指定实验目录")
                with self.server.store_lock:
                    document = self.server.store.get_run(run_match[1])
                return self._json(dict(**document, read_only=True), head=head)
            if path == "/api/asset":
                episode, relative = query.get("episode", [""])[0], query.get("path", [""])[0]
                with self.server.store_lock:
                    asset = self.server.store.resolve_asset(episode, relative)
                return self._file(asset, head)
            if re.match(r"^/assets/[0-9a-f]{16}/", path):
                parts = path.split("/", 3)
                if len(parts) != 4:
                    raise ValueError("资产路径不完整")
                with self.server.store_lock:
                    asset = self.server.store.resolve_asset(parts[2], parts[3])
                return self._file(asset, head)
            match = re.fullmatch(r"/api/episodes/([^/]+)(?:/(frames|potential)/(\d+))?", path)
            if match:
                episode, action, index = match.groups()
                with self.server.store_lock:
                    self.server.store.refresh(episode)
                    document = (self.server.store.get_manifest(episode) if action is None else
                                self.server.store.get_frame(episode, int(index)))
                if action is not None:
                    if action == "potential":
                        if document.get("dataset") == "navsim":
                            return self._json(dict(available=False, reason="NAVSIM只读查看器不重新运行规划核心或重建评分势场；请查看源帧归档BEV与规划证据"), head=head)
                        try:
                            from .geometry import build_potential
                            document = build_potential(document, engine_root=self.server.engine_root)
                        except ImportError as exc:
                            document = dict(available=False, reason="势场依赖不可用：" + str(exc))
                        return self._json(document, head=head)
                    try:
                        from .geometry import enrich_frame
                        document = enrich_frame(document, engine_root=None if document.get("dataset") == "navsim" else self.server.engine_root)
                        if document.get("dataset") == "navsim":
                            document = library_wire_frame(document)
                            document["map"] = None  # Rendered geometry already contains the source map.
                            vis = document.get("visualization", {})
                            vis["trajectory_provenance"] = dict(kind="archived_plan" if document["index"] == 0 else "recorded_future_execution",
                                title="NAVSIM官方评分执行；独立4秒锚点")
                            vis["corridor_note"] = "源锚点局部坐标；源规划为车身中心，提交与官方轨迹为后轴。未来为评分回放，无重新规划或未来相机图。"
                    except ImportError as exc:
                        document.setdefault("warnings", []).append("三维几何依赖不可用：" + str(exc))
                return self._json(self.server.decorate(episode, document), head=head)
            relative = "index.html" if path == "/" else "memory.html" if path == "/memory" else path.lstrip("/")
            file_path = (WEB / relative).resolve()
            if not file_path.is_relative_to(WEB.resolve()):
                raise PermissionError("静态文件路径越界")
            return self._file(file_path, head)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except (PermissionError, ValueError) as exc:
            return self._json(dict(error=str(exc)), 400, head)
        except (FileNotFoundError, KeyError, IndexError) as exc:
            return self._json(dict(error=str(exc)), 404, head)
        except Exception as exc:
            self.log_error("%s: %s", type(exc).__name__, exc)
            return self._json(dict(error=type(exc).__name__ + ": " + str(exc)), 500, head)

    def _file(self, path, head=False):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("资产尚未产生或不存在")
        size = path.stat().st_size
        first, last, partial = 0, size - 1, False
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if match and size and (match[1] or match[2]):
                if not match[1]:
                    first = max(0, size - int(match[2]))
                else:
                    first = int(match[1])
                    last = min(last, int(match[2])) if match[2] else last
                partial = first <= last and first < size
            if not partial:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        length = max(0, last - first + 1)
        self.send_response(206 if partial else 200)
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript"
        if content_type.startswith("text/"):
            content_type += "; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if partial:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.end_headers()
        if head:
            return
        with path.open("rb") as stream:
            stream.seek(first)
            while length:
                block = stream.read(min(length, 1024 * 256))
                if not block:
                    break
                self.wfile.write(block)
                length -= len(block)

    def log_message(self, fmt, *args):
        # Console keeps startup and errors useful; polling never produces 10Hz logs.
        if args and isinstance(args[0], str) and "/api/" in args[0] and str(args[1]) in ("200", "206"):
            return
        super().log_message(fmt, *args)


def main(argv=None):
    parser = argparse.ArgumentParser(description="EP-Driver独立只读展示：运行产物实时跟随和离线回放")
    parser.add_argument("--root", type=Path, help="单轮结果目录或包含多轮的结果根目录")
    parser.add_argument("--memory-root", type=Path, help="离线记忆发布库，只读浏览manifest登记的经验及资产")
    parser.add_argument("--engine-root", type=Path, help="对应版本EP-Driver/CARLA仓库，仅只读调用几何与势场函数")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="自动在本机浏览器打开展示页")
    parser.add_argument("--follow", action="store_true", help="打开页面时默认跟随最新产物")
    parser.add_argument("--cache-dir", type=Path, help="兼容视频副本目录，默认位于展示程序自己的.cache/video")
    parser.add_argument("--ffmpeg", help="可选FFmpeg可执行文件路径；默认从PATH查找")
    args = parser.parse_args(argv)
    if args.root is None and args.memory_root is None:
        parser.error("请至少指定 --root 实验目录或 --memory-root 离线记忆库目录")
    if args.root is not None and not args.root.is_dir():
        parser.error("--root必须是已存在的结果目录（可在实验刚开始、文件仍增长时打开）")
    if args.memory_root is not None and not args.memory_root.is_dir():
        parser.error("--memory-root必须是已存在的离线记忆库目录")
    try:
        server = ViewerServer((args.host, args.port), args.root, args.engine_root, args.cache_dir, args.ffmpeg,
                              memory_root=args.memory_root)
    except (OSError, ValueError) as exc:
        parser.error(f"无法启动 {args.host}:{args.port}；请检查输入目录及库清单，端口被占用时可用--port修改。{exc}")
    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    page = '/memory' if args.root is None else '/'
    url = f"http://{browser_host}:{server.server_port}" + page + ("?follow=1" if args.follow else "")
    print(f"EP Agent Viewer {__version__}", flush=True)
    if args.root is not None:
        print(f"只读结果目录：{args.root.resolve()}", flush=True)
    if args.memory_root is not None:
        print(f"只读记忆库：{args.memory_root.resolve()}", flush=True)
    print(f"打开：{url}", flush=True)
    print("展示程序不连接CARLA、不推进仿真、不调用模型、不修改运行产物。Ctrl+C退出。", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
