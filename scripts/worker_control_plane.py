import json
import os
import shutil
import socket
import subprocess
import threading
import time
from typing import Callable, Optional
from urllib import error, request


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_base_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    value = raw.strip().rstrip("/")
    if not value:
        return None
    if "://" not in value:
        value = f"http://{value}"
    return value


def detect_gpu_type() -> Optional[str]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


class LinguaWorkerControlPlane:
    def __init__(
        self,
        *,
        internal_port: int,
        profile: str,
        metrics_provider: Callable[[], dict],
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.internal_port = int(internal_port)
        self.profile = profile
        self.metrics_provider = metrics_provider
        self.log_fn = log_fn or (lambda message: print(message, flush=True))

        self.worker_type = os.getenv("LINGUA_WORKER_TYPE", "musetalk").strip() or "musetalk"
        self.capacity = max(1, _env_int("LINGUA_WORKER_DEFAULT_CAPACITY", 1))
        self.instance_id = (
            os.getenv("VAST_INSTANCE_ID")
            or os.getenv("CONTAINER_ID")
            or ""
        ).strip()
        self.worker_id = (
            os.getenv("WORKER_ID")
            or f"{self.worker_type}-{self.instance_id or socket.gethostname()}"
        ).strip()
        self.region = (os.getenv("LINGUA_WORKER_REGION") or os.getenv("VAST_REGION") or "").strip()
        self.version = (os.getenv("LINGUA_WORKER_BUILD") or os.getenv("GIT_COMMIT") or "").strip()
        self.public_ip = (
            os.getenv("LINGUA_WORKER_PUBLIC_IP")
            or os.getenv("VAST_PUBLIC_IP")
            or os.getenv("PUBLIC_IPADDR")
            or ""
        ).strip()
        self.public_port = (
            os.getenv("LINGUA_WORKER_PUBLIC_PORT")
            or os.getenv("VAST_PUBLIC_PORT")
            or os.getenv(f"VAST_TCP_PORT_{self.internal_port}")
            or ""
        ).strip()
        self.scheme = (os.getenv("LINGUA_WORKER_PUBLIC_SCHEME") or "http").strip() or "http"
        self.base_url = self._discover_public_base_url()
        self.gpu_type = (os.getenv("LINGUA_WORKER_GPU_TYPE") or detect_gpu_type() or "").strip()

        control_plane_base = _normalize_base_url(os.getenv("LINGUA_CONTROL_PLANE_BASE_URL"))
        self.register_url = _normalize_base_url(os.getenv("LINGUA_WORKER_REGISTER_URL"))
        self.heartbeat_url = _normalize_base_url(os.getenv("LINGUA_WORKER_HEARTBEAT_URL"))
        if control_plane_base:
            self.register_url = self.register_url or f"{control_plane_base}/api/runtime/workers/register"
            self.heartbeat_url = self.heartbeat_url or f"{control_plane_base}/api/runtime/workers/heartbeat"

        self.token = (os.getenv("LINGUA_WORKER_TOKEN") or "").strip()
        self.heartbeat_interval_seconds = max(5, _env_int("LINGUA_HEARTBEAT_INTERVAL_SECONDS", 20))
        self.busy_heartbeat_interval_seconds = max(
            5,
            _env_int("LINGUA_BUSY_HEARTBEAT_INTERVAL_SECONDS", 5),
        )
        self.registration_retry_seconds = max(2, _env_int("LINGUA_REGISTER_RETRY_SECONDS", 5))
        self.registration_timeout_seconds = max(
            2,
            _env_int("LINGUA_REGISTER_TIMEOUT_SECONDS", 10),
        )
        self.heartbeat_timeout_seconds = max(
            2,
            _env_int("LINGUA_HEARTBEAT_TIMEOUT_SECONDS", 10),
        )
        self.drain_timeout_seconds = max(1, _env_int("LINGUA_DRAIN_TIMEOUT_SECONDS", 300))

        self.control_plane_requested = any(
            [
                control_plane_base,
                self.register_url,
                self.heartbeat_url,
                self.token,
            ]
        )
        self.control_plane_configured = all(
            [
                self.register_url,
                self.heartbeat_url,
                self.token,
                self.base_url,
            ]
        )

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._registered_event = threading.Event()
        self._local_ready = False
        self._draining = False
        self._registration_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_register_error: Optional[str] = None
        self._last_heartbeat_error: Optional[str] = None
        self._last_register_response: Optional[str] = None
        self._last_heartbeat_response: Optional[str] = None
        self._last_register_success_at: Optional[float] = None
        self._last_heartbeat_at: Optional[float] = None
        self._last_drain_reason: Optional[str] = None

    def _log(self, message: str) -> None:
        self.log_fn(f"[worker-control] {message}")

    def _discover_public_base_url(self) -> Optional[str]:
        explicit = _normalize_base_url(
            os.getenv("LINGUA_WORKER_BASE_URL") or os.getenv("WORKER_PUBLIC_BASE_URL")
        )
        if explicit:
            return explicit
        if self.public_ip and self.public_port:
            return f"{self.scheme}://{self.public_ip}:{self.public_port}"
        return None

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "X-Worker-Token": self.token,
        }

    def _active_metrics(self) -> dict:
        try:
            return self.metrics_provider() or {}
        except Exception as exc:
            return {"metrics_error": str(exc)}

    def mark_local_ready(self) -> None:
        with self._lock:
            self._local_ready = True
        self._log("Local MuseTalk runtime is ready")

    def is_registered(self) -> bool:
        return self._registered_event.is_set()

    def is_draining(self) -> bool:
        with self._lock:
            return self._draining

    def current_status(self) -> str:
        with self._lock:
            local_ready = self._local_ready
            draining = self._draining
        if draining:
            return "draining"
        if not local_ready:
            return "booting"
        if self.control_plane_requested and not self.control_plane_configured:
            return "unhealthy"
        if self.control_plane_requested and not self.is_registered():
            return "registering"
        return "healthy"

    def ready_for_health(self) -> bool:
        with self._lock:
            local_ready = self._local_ready
            draining = self._draining
        if not local_ready:
            return False
        if draining:
            return True
        if not self.control_plane_requested:
            return True
        if not self.control_plane_configured:
            return False
        return self.is_registered()

    def accepting_new_sessions(self) -> bool:
        return self.ready_for_health() and not self.is_draining()

    def _payload(self) -> dict:
        metrics = self._active_metrics()
        payload = {
            "worker_type": self.worker_type,
            "worker_id": self.worker_id,
            "instance_id": self.instance_id,
            "vast_instance_id": self.instance_id,
            "base_url": self.base_url,
            "endpoint_url": self.base_url,
            "capacity": self.capacity,
            "max_concurrency": self.capacity,
            "status": self.current_status(),
            "internal_port": self.internal_port,
            "metadata": {
                **metrics,
                "profile": self.profile,
                "draining": self.is_draining(),
                "build": self.version or None,
            },
        }
        if self.region:
            payload["region"] = self.region
        if self.gpu_type:
            payload["gpu_type"] = self.gpu_type
        if self.public_ip:
            payload["public_ip"] = self.public_ip
        if self.public_port:
            try:
                payload["public_port"] = int(self.public_port)
            except ValueError:
                payload["public_port"] = self.public_port
        return payload

    def _post_json(self, url: str, payload: dict, timeout_seconds: int) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, method="POST", headers=self._headers())
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace").strip()
                return raw
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"{exc.code} {exc.reason}: {response_body[:300]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

    def begin_draining(self, reason: str = "manual") -> None:
        with self._lock:
            if self._draining:
                return
            self._draining = True
            self._last_drain_reason = reason
        self._log(f"Entering drain mode (reason={reason})")
        if self.control_plane_configured and self.is_registered():
            try:
                response = self._post_json(
                    self.heartbeat_url,
                    self._payload(),
                    timeout_seconds=self.heartbeat_timeout_seconds,
                )
                with self._lock:
                    self._last_heartbeat_at = time.time()
                    self._last_heartbeat_response = response[:500]
                    self._last_heartbeat_error = None
                self._log("Sent immediate draining heartbeat")
            except Exception as exc:
                with self._lock:
                    self._last_heartbeat_error = str(exc)
                self._log(f"Immediate draining heartbeat failed: {exc}")

    def wait_for_idle(self, timeout_seconds: Optional[int] = None) -> bool:
        timeout = self.drain_timeout_seconds if timeout_seconds is None else max(0, int(timeout_seconds))
        deadline = time.time() + timeout
        while time.time() <= deadline and not self._stop_event.is_set():
            metrics = self._active_metrics()
            active_requests = int(metrics.get("active_requests", 0) or 0)
            active_sessions = int(metrics.get("active_sessions_local", 0) or 0)
            queue_depth = int(metrics.get("queue_depth", 0) or 0)
            if active_requests <= 0 and active_sessions <= 0 and queue_depth <= 0:
                return True
            time.sleep(1.0)
        return False

    def start(self) -> None:
        if not self.control_plane_requested:
            self._log("Lingua control plane integration disabled (no env vars present)")
            return

        if not self.control_plane_configured:
            missing = []
            if not self.register_url:
                missing.append("LINGUA_WORKER_REGISTER_URL")
            if not self.heartbeat_url:
                missing.append("LINGUA_WORKER_HEARTBEAT_URL")
            if not self.token:
                missing.append("LINGUA_WORKER_TOKEN")
            if not self.base_url:
                missing.append("public base URL discovery")
            self._last_register_error = f"Missing control-plane configuration: {', '.join(missing)}"
            self._log(self._last_register_error)
            return

        self._log(
            f"Configured worker_id={self.worker_id} instance_id={self.instance_id or '-'} "
            f"base_url={self.base_url}"
        )
        self._registration_thread = threading.Thread(
            target=self._registration_loop,
            name="lingua-register",
            daemon=True,
        )
        self._registration_thread.start()

    def _registration_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                local_ready = self._local_ready
            if not local_ready:
                time.sleep(1.0)
                continue

            try:
                response = self._post_json(
                    self.register_url,
                    self._payload(),
                    timeout_seconds=self.registration_timeout_seconds,
                )
                with self._lock:
                    self._last_register_error = None
                    self._last_register_response = response[:500]
                    self._last_register_success_at = time.time()
                self._registered_event.set()
                self._log("Worker registration succeeded")
                self._start_heartbeat_loop()
                return
            except Exception as exc:
                with self._lock:
                    self._last_register_error = str(exc)
                self._log(
                    f"Registration failed; retrying in {self.registration_retry_seconds}s: {exc}"
                )
                self._stop_event.wait(self.registration_retry_seconds)

    def _start_heartbeat_loop(self) -> None:
        if self._heartbeat_thread is not None:
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="lingua-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                response = self._post_json(
                    self.heartbeat_url,
                    self._payload(),
                    timeout_seconds=self.heartbeat_timeout_seconds,
                )
                with self._lock:
                    self._last_heartbeat_at = time.time()
                    self._last_heartbeat_error = None
                    self._last_heartbeat_response = response[:500]
            except Exception as exc:
                with self._lock:
                    self._last_heartbeat_error = str(exc)
                self._log(f"Heartbeat failed: {exc}")

            metrics = self._active_metrics()
            active_requests = int(metrics.get("active_requests", 0) or 0)
            active_sessions = int(metrics.get("active_sessions_local", 0) or 0)
            interval = self.heartbeat_interval_seconds
            if self.is_draining() or active_requests > 0 or active_sessions > 0:
                interval = self.busy_heartbeat_interval_seconds
            self._stop_event.wait(interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._registration_thread is not None:
            self._registration_thread.join(timeout=2.0)
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)

    def state_snapshot(self) -> dict:
        metrics = self._active_metrics()
        with self._lock:
            return {
                "worker_id": self.worker_id,
                "worker_type": self.worker_type,
                "instance_id": self.instance_id,
                "base_url": self.base_url,
                "public_ip": self.public_ip or None,
                "public_port": self.public_port or None,
                "internal_port": self.internal_port,
                "capacity": self.capacity,
                "status": self.current_status(),
                "draining": self._draining,
                "accepting_new_sessions": self.accepting_new_sessions(),
                "local_ready": self._local_ready,
                "control_plane_requested": self.control_plane_requested,
                "control_plane_configured": self.control_plane_configured,
                "registered": self._registered_event.is_set(),
                "register_url": self.register_url,
                "heartbeat_url": self.heartbeat_url,
                "last_register_error": self._last_register_error,
                "last_heartbeat_error": self._last_heartbeat_error,
                "last_register_response": self._last_register_response,
                "last_heartbeat_response": self._last_heartbeat_response,
                "last_register_success_at": self._last_register_success_at,
                "last_heartbeat_at": self._last_heartbeat_at,
                "last_drain_reason": self._last_drain_reason,
                "metrics": metrics,
            }
