import os
import sys
from typing import Iterable


_EARLY_SUMMARY = None
_RUNTIME_SUMMARY = None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None, minimum: int = 1) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _parse_cpu_list(raw: str) -> list[int]:
    cpus: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                start, end = end, start
            cpus.extend(range(start, end + 1))
            continue
        cpus.append(int(token))
    return sorted(set(cpus))


def _cpus_for_numa_node(node_index: int) -> list[int]:
    cpulist_path = f"/sys/devices/system/node/node{node_index}/cpulist"
    with open(cpulist_path, "r", encoding="utf-8") as handle:
        return _parse_cpu_list(handle.read().strip())


def _format_cpu_list(cpus: Iterable[int], max_items: int = 12) -> str:
    values = list(cpus)
    if len(values) <= max_items:
        return ",".join(str(cpu) for cpu in values)
    shown = ",".join(str(cpu) for cpu in values[:max_items])
    return f"{shown},... ({len(values)} total)"


def _recommended_threads() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 4:
        return cpu_count
    if cpu_count <= 16:
        return 4
    return 8


def _cpu_tuning_enabled() -> bool:
    return _env_bool("MUSETALK_CPU_TUNING", False)


def _build_summary() -> dict:
    thread_cap = _env_int("MUSETALK_CPU_THREADS", default=_recommended_threads())
    interop_threads = _env_int(
        "MUSETALK_CPU_INTEROP_THREADS",
        default=max(1, thread_cap // 2),
    )
    cv2_threads = _env_int("MUSETALK_CPU_CV2_THREADS", default=1)
    affinity_raw = (os.getenv("MUSETALK_CPU_AFFINITY") or "").strip()
    numa_node = _env_int("MUSETALK_CPU_NUMA_NODE", default=None, minimum=0)
    return {
        "enabled": True,
        "thread_cap": int(thread_cap),
        "interop_threads": int(interop_threads),
        "cv2_threads": int(cv2_threads),
        "affinity_raw": affinity_raw,
        "numa_node": numa_node,
    }


def apply_cpu_tuning_early(process_label: str = "process") -> dict:
    global _EARLY_SUMMARY
    if _EARLY_SUMMARY is not None:
        return _EARLY_SUMMARY
    if not _cpu_tuning_enabled():
        _EARLY_SUMMARY = {"enabled": False}
        return _EARLY_SUMMARY

    summary = _build_summary()
    thread_cap = summary["thread_cap"]
    interop_threads = summary["interop_threads"]
    cv2_threads = summary["cv2_threads"]

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(thread_cap)
    os.environ["OPENCV_THREAD_COUNT"] = str(cv2_threads)
    os.environ.setdefault("OMP_PROC_BIND", "true")

    affinity_source = None
    affinity_error = None
    affinity_cpus: list[int] = []
    try:
        if summary["numa_node"] is not None:
            affinity_cpus = _cpus_for_numa_node(summary["numa_node"])
            affinity_source = f"numa:{summary['numa_node']}"
        elif summary["affinity_raw"]:
            affinity_cpus = _parse_cpu_list(summary["affinity_raw"])
            affinity_source = "manual"

        if affinity_cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, affinity_cpus)
    except Exception as exc:
        affinity_error = str(exc)
        affinity_cpus = []
        affinity_source = None

    summary.update(
        {
            "affinity_source": affinity_source,
            "affinity_cpus": affinity_cpus,
            "affinity_error": affinity_error,
        }
    )
    _EARLY_SUMMARY = summary

    message = (
        f"🧠 CPU tuning enabled for {process_label}: "
        f"threads={thread_cap}, interop={interop_threads}, cv2={cv2_threads}"
    )
    if affinity_cpus:
        message += f", affinity={_format_cpu_list(affinity_cpus)}"
    elif affinity_error:
        message += f", affinity_error={affinity_error}"
    print(message)
    return _EARLY_SUMMARY


def apply_cpu_tuning_runtime(process_label: str = "process") -> dict:
    global _RUNTIME_SUMMARY
    if _RUNTIME_SUMMARY is not None:
        return _RUNTIME_SUMMARY
    if not _cpu_tuning_enabled():
        _RUNTIME_SUMMARY = {"enabled": False}
        return _RUNTIME_SUMMARY

    summary = dict(apply_cpu_tuning_early(process_label))
    torch_status = "not-imported"
    cv2_status = "not-imported"

    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            torch_module.set_num_threads(summary["thread_cap"])
            torch_status = f"threads={torch_module.get_num_threads()}"
        except Exception as exc:
            torch_status = f"thread_error={exc}"
        try:
            torch_module.set_num_interop_threads(summary["interop_threads"])
            interop_value = torch_module.get_num_interop_threads()
            torch_status += f", interop={interop_value}"
        except Exception as exc:
            torch_status += f", interop_error={exc}"

    cv2_module = sys.modules.get("cv2")
    if cv2_module is not None:
        try:
            cv2_module.setNumThreads(summary["cv2_threads"])
            cv2_status = f"threads={cv2_module.getNumThreads()}"
        except Exception as exc:
            cv2_status = f"error={exc}"

    summary.update(
        {
            "torch_status": torch_status,
            "cv2_status": cv2_status,
        }
    )
    _RUNTIME_SUMMARY = summary
    print(
        f"🧠 CPU runtime tuning active for {process_label}: "
        f"torch={torch_status}, cv2={cv2_status}"
    )
    return _RUNTIME_SUMMARY
