"""Local mission control with background runs and an inspectable HTTP API.

The browser has no model credentials or direct filesystem access. Each action
uses the same checkpointed Collective as the CLI, and restarting this server
leaves existing runs available to resume.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from .config import Settings
from .control import RunControl
from .runtime import Collective, RunBusyError


@dataclass
class ManagedRun:
    collective: Collective
    control: RunControl | None = None
    thread: threading.Thread | None = None
    error: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


class RunManager:
    """Keep control gates in memory; all collective data lives on disk."""

    def __init__(self, runs_dir: str | Path = "runs", *, models=None):
        self.directory = Path(runs_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models = models
        self._runs: dict[str, ManagedRun] = {}
        self._lock = threading.RLock()

    def list_runs(self) -> list[dict]:
        runs = []
        for path in sorted(self.directory.glob("*/run.json"), reverse=True):
            if not path.resolve().is_relative_to(self.directory):
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("version") not in (2, 4):
                    continue
                run_id = path.parent.name
                managed = self._runs.get(run_id)
                runs.append(
                    {
                        **manifest,
                        "run_id": run_id,
                        "running": bool(managed and managed.running),
                        "control": managed.control.snapshot()
                        if managed and managed.control
                        else None,
                    }
                )
            except (OSError, ValueError, TypeError):
                continue
        return runs

    def get(self, run_id: str) -> ManagedRun:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise ValueError("Invalid run ID")
        directory = (self.directory / run_id).resolve()
        if not directory.is_relative_to(self.directory):
            raise ValueError("Run leaves the runs directory")
        with self._lock:
            if run_id not in self._runs:
                if not (directory / "run.json").is_file():
                    raise FileNotFoundError("Run does not exist")
                self._runs[run_id] = ManagedRun(Collective.open(directory, models=self.models))
            return self._runs[run_id]

    def create(self, goal: str, settings: dict | None = None) -> dict:
        if not isinstance(goal, str):
            raise ValueError("The goal must be text")
        collective = Collective.create(
            goal, self.directory, Settings.model_validate(settings or {}), models=self.models
        )
        with self._lock:
            self._runs[collective.manifest["run_id"]] = ManagedRun(collective)
        return self.snapshot(collective.manifest["run_id"])

    @staticmethod
    def read_events(directory: Path, cursor: int = 0, limit: int = 500) -> tuple[list, int]:
        """Byte cursors never lose a partially written JSONL event."""
        path = directory / "events.jsonl"
        if not path.is_file():
            return [], 0
        events = []
        with path.open("rb") as stream:
            size = path.stat().st_size
            if cursor < 0 or cursor > size:
                cursor = 0
            stream.seek(cursor)
            for _ in range(limit):
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    break
                cursor = stream.tell()
                try:
                    events.append(json.loads(line))
                except (ValueError, UnicodeError):
                    continue
        return events, cursor

    def snapshot(self, run_id: str, cursor: int = 0) -> dict:
        managed = self.get(run_id)
        run = managed.collective
        events, next_cursor = self.read_events(run.directory, cursor)
        artifacts = []
        for path in run.workspace.list():
            try:
                file = run.workspace.path(path)
                artifacts.append({"path": path, "bytes": file.stat().st_size})
            except FileNotFoundError:
                continue
        return {
            "manifest": dict(run.manifest),
            "state": run.state(include_workers=True),
            "running": managed.running,
            "control": managed.control.snapshot() if managed.control else None,
            "error": managed.error or run.manifest.get("last_error"),
            "events": events,
            "cursor": next_cursor,
            "artifacts": artifacts,
        }

    def control(self, run_id: str, action: str, *, retry: bool = False) -> dict:
        managed = self.get(run_id)
        with managed.lock:
            if action not in {"start", "pause", "resume", "step", "stop"}:
                raise ValueError("Unknown control action")
            if not managed.running and action in {"start", "resume", "step"}:
                managed.control = RunControl(start_paused=action == "step")
                managed.error = None

                def work():
                    try:
                        managed.collective.run(retry=retry, control=managed.control)
                    except Exception as exc:
                        managed.error = f"{type(exc).__name__}: {exc}"

                managed.thread = threading.Thread(
                    target=work, name=f"collective-{run_id}", daemon=True
                )
                if action == "step":
                    managed.control.step()
                managed.thread.start()
            elif managed.running:
                if action == "start":
                    raise RunBusyError("This run is already active")
                getattr(managed.control, action)()
            elif action in {"pause", "stop"}:
                raise RunBusyError("This run is idle")
        return self.snapshot(run_id)

    def admin(self, run_id: str, action: str, fields: dict) -> dict:
        managed = self.get(run_id)
        with managed.lock:
            managed.collective.admin(action, control=managed.control, **fields)
            managed.control = RunControl(start_paused=True)
        return self.snapshot(run_id)

    def close(self):
        for managed in list(self._runs.values()):
            if managed.running and managed.control:
                managed.control.stop()


class ControlRoomServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], manager: RunManager):
        self.manager = manager
        super().__init__(address, ControlRoomHandler)

    def server_close(self):
        self.manager.close()
        super().server_close()


class ControlRoomHandler(BaseHTTPRequestHandler):
    server: ControlRoomServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        # Model and graph activity is available in the debug feed.
        pass

    def send_data(self, data: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: dict | list, status: int = 200):
        self.send_data(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def body(self) -> dict:
        if getattr(self, "_request_body", None) is not None:
            return self._request_body
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid request size") from exc
        if not 0 < length <= 4_200_000:
            raise ValueError("Expected a JSON body smaller than 4 MB")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value

    def check_origin(self):
        # Local browser applications must not permit another site to issue actions.
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if origin and urlsplit(origin).netloc != host:
            raise PermissionError("This action must come from this control room")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise PermissionError("Cross-site requests are not allowed")

    def route(self, method: str):
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        parts = path.strip("/").split("/")
        query = parse_qs(parsed.query)
        manager = self.server.manager
        if method == "GET" and path in {"/", "/index.html", "/app.js", "/style.css"}:
            name = "index.html" if path == "/" else path[1:]
            data = files("kollektiv").joinpath("web", name).read_bytes()
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            return self.send_data(data, content_type + "; charset=utf-8")
        if method == "GET" and path == "/api/bootstrap":
            return self.send_json(
                {"settings": Settings().model_dump(), "runs": manager.list_runs(), "version": 4}
            )
        if method == "GET" and path == "/api/health":
            run_id = query.get("run", [""])[0]
            settings = manager.get(run_id).collective.settings if run_id else Settings()
            try:
                response = httpx.get(f"{settings.base_url}/api/tags", timeout=4)
                response.raise_for_status()
                models = [m["name"] for m in response.json().get("models", [])]
                return self.send_json(
                    {"connected": True, "model": settings.model, "models": models}
                )
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                return self.send_json(
                    {"connected": False, "model": settings.model, "error": str(exc)}
                )
        if method == "GET" and path == "/api/runs":
            return self.send_json({"runs": manager.list_runs()})
        if method == "POST" and path == "/api/runs":
            data = self.body()
            return self.send_json(manager.create(data.get("goal", ""), data.get("settings")), 201)
        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            run_id = parts[2]
            if method == "GET" and len(parts) == 3:
                cursor = int(query.get("cursor", ["0"])[0])
                return self.send_json(manager.snapshot(run_id, cursor))
            if method == "GET" and len(parts) == 4 and parts[3] == "artifact":
                name = query.get("path", [""])[0]
                workspace = manager.get(run_id).collective.workspace
                return self.send_json({"path": name, "content": workspace.read(name)})
            if method == "POST" and len(parts) == 4:
                data = self.body()
                action = data.pop("action", "")
                if parts[3] == "control":
                    return self.send_json(
                        manager.control(run_id, action, retry=bool(data.get("retry", False)))
                    )
                if parts[3] == "admin":
                    return self.send_json(manager.admin(run_id, action, data))
        return self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def handle_request(self, method: str):
        self._request_body = None
        try:
            if method == "POST":
                # Rejected requests may have unread bodies. Do not reuse that socket.
                self.close_connection = True
                self._request_body = self.body()
                self.check_origin()
            self.route(method)
        except RunBusyError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except (ValueError, TypeError, KeyError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self):
        self.handle_request("GET")

    def do_POST(self):
        self.handle_request("POST")


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    runs_dir: str | Path = "runs",
    open_browser: bool = False,
) -> None:
    server = ControlRoomServer((host, port), RunManager(runs_dir))
    url = f"http://{host}:{server.server_port}"
    print(f"Kollektiv 4 control room: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Stopping control room. Run progress remains checkpointed.", flush=True)
    finally:
        server.server_close()
