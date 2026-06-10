from __future__ import annotations

import time
from typing import Any, Dict, List

from .llm import llm_runtime_options
from .utils import unique_strings


class LLMBaselineRuntime:
    def __init__(
        self,
        *,
        enabled: bool,
        timeout_seconds: int,
        total_timeout_seconds: int,
        stage_timeout_seconds: int,
        max_docs: int,
        skip_tasks: bool,
    ) -> None:
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self.stage_timeout_seconds = stage_timeout_seconds
        self.max_docs = max_docs
        self.skip_tasks = skip_tasks
        self.started = time.time()
        self.budget_exhausted = False
        self._stages: Dict[str, LLMStageRuntime] = {}

    def limit_doc_ids(self, doc_ids: List[str]) -> List[str]:
        if not self.enabled:
            return []
        if self.max_docs <= 0:
            return list(doc_ids)
        return list(doc_ids[: self.max_docs])

    def stage(self, name: str) -> "LLMStageRuntime":
        stage = self._stages.get(name)
        if stage is None:
            stage = LLMStageRuntime(self, name)
            self._stages[name] = stage
        return stage

    def elapsed_ms(self) -> float:
        return round((time.time() - self.started) * 1000, 3)

    def budget_remaining(self) -> bool:
        if not self.enabled:
            return False
        if self.budget_exhausted:
            return False
        if time.time() - self.started > self.total_timeout_seconds:
            self.budget_exhausted = True
            return False
        return True

    def summary(self) -> Dict[str, Any]:
        stages = {name: stage.summary() for name, stage in self._stages.items()}
        return {
            "schema": "llm_runtime_summary.v1",
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "stage_timeout_seconds": self.stage_timeout_seconds,
            "max_docs": self.max_docs,
            "skip_tasks": self.skip_tasks,
            "stage_summary": stages,
            "total_llm_duration_ms": round(sum(float(stage.get("llm_duration_ms") or 0.0) for stage in stages.values()), 3),
            "total_llm_call_count": sum(int(stage.get("call_count") or 0) for stage in stages.values()),
            "timeout_count": sum(int(stage.get("timeout_count") or 0) for stage in stages.values()),
            "hard_timeout_count": sum(int(stage.get("hard_timeout_count") or 0) for stage in stages.values()),
            "slow_call_count": sum(int(stage.get("slow_call_count") or 0) for stage in stages.values()),
            "fallback_count": sum(int(stage.get("fallback_count") or 0) for stage in stages.values()),
            "budget_exhausted": self.budget_exhausted,
            "elapsed_ms": self.elapsed_ms(),
        }


class LLMStageRuntime:
    def __init__(self, runtime: LLMBaselineRuntime, name: str) -> None:
        self.runtime = runtime
        self.name = name
        self.status = "pending"
        self.reason = ""
        self.warnings: List[str] = []
        self.call_count = 0
        self.timeout_count = 0
        self.hard_timeout_count = 0
        self.slow_call_count = 0
        self._call_durations: List[float] = []
        self.fallback_count = 0
        self.llm_duration_ms = 0.0
        self.started = 0.0
        self.duration_ms = 0.0
        self._ctx = None

    def __enter__(self) -> "LLMStageRuntime":
        self.started = time.time()
        if not self.runtime.enabled:
            self.status = "skipped"
            self.reason = "llm_disabled"
            return self
        if not self.runtime.budget_remaining():
            self.status = "skipped"
            self.reason = "baseline_llm_budget_exhausted"
            self.warnings.append(self.reason)
            return self
        self.status = "completed"
        self._ctx = llm_runtime_options(
            timeout_seconds=self.runtime.timeout_seconds,
            operation="quality_baseline",
            stage=self.name,
            event_collector=self.record_event,
        )
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc, tb)
        self.duration_ms = round((time.time() - self.started) * 1000, 3) if self.started else 0.0
        if self.status == "completed":
            if exc_type is not None:
                self.status = "timeout" if getattr(exc, "error_type", "") == "request_timeout" else "failed"
                self.warnings.append(f"{self.name}_{self.status}")
            elif self.duration_ms > self.runtime.stage_timeout_seconds * 1000:
                self.status = "timeout" if self.call_count == 0 or (self.timeout_count and self.timeout_count >= self.call_count) else "partial"
                self.reason = "stage_timeout" if self.status == "timeout" else "stage_budget_exceeded"
                self.warnings.append("stage_timeout")
            elif self.timeout_count and self.timeout_count >= max(1, self.call_count):
                self.status = "timeout"
                self.warnings.append(f"{self.name}_timeout")
            elif self.timeout_count or self.fallback_count or self.warnings:
                self.status = "partial"
        if time.time() - self.runtime.started > self.runtime.total_timeout_seconds:
            self.runtime.budget_exhausted = True

    @property
    def allowed(self) -> bool:
        return self.status not in {"skipped"}

    def can_continue(self) -> bool:
        if not self.allowed:
            return False
        if not self.runtime.budget_remaining():
            self.status = "skipped"
            self.reason = "baseline_llm_budget_exhausted"
            self.warnings.append(self.reason)
            return False
        if self.started and time.time() - self.started > self.runtime.stage_timeout_seconds:
            self.status = "timeout"
            self.reason = "stage_timeout"
            self.warnings.append("stage_timeout")
            return False
        return True

    def mark_fallback(self, reason: str = "fallback") -> None:
        self.fallback_count += 1
        if reason:
            self.warnings.append(reason)

    def mark_warning(self, warning: str) -> None:
        if warning:
            self.warnings.append(warning)

    def record_event(self, event: Dict[str, Any]) -> None:
        self.call_count += 1
        duration_ms = float(event.get("duration_ms") or 0.0)
        timeout_ms = float(event.get("timeout_seconds") or self.runtime.timeout_seconds or 0) * 1000
        self._call_durations.append(duration_ms)
        self.llm_duration_ms += duration_ms
        if timeout_ms and duration_ms >= timeout_ms * 0.8:
            self.slow_call_count += 1
        if event.get("status") == "timeout" or event.get("error_type") == "request_timeout":
            self.timeout_count += 1
            self.hard_timeout_count += 1
            self.warnings.append("request_timeout")
        elif event.get("status") == "failed":
            self.warnings.append(str(event.get("error_type") or "llm_failed"))

    def record_call(self, duration_ms: float) -> None:
        self.record_event(
            {
                "duration_ms": duration_ms,
                "timeout_seconds": self.runtime.timeout_seconds,
                "status": "ok",
            }
        )

    def record_timeout(self, *, hard: bool = True) -> None:
        self.timeout_count += 1
        if hard:
            self.hard_timeout_count += 1
        self.warnings.append("request_timeout")

    def record_fallback(self, reason: str = "fallback") -> None:
        self.mark_fallback(reason)

    def summary(self) -> Dict[str, Any]:
        status = self.status if self.status != "pending" else "skipped"
        reason = self.reason or ("not_started" if self.status == "pending" else "")
        return {
            "schema": "llm_stage_runtime.v1",
            "stage": self.name,
            "status": status,
            "reason": reason,
            "duration_ms": self.duration_ms,
            "llm_duration_ms": round(self.llm_duration_ms, 3),
            "call_count": self.call_count,
            "avg_call_duration_ms": round(sum(self._call_durations) / max(1, len(self._call_durations)), 3),
            "slow_call_count": self.slow_call_count,
            "fallback_count": self.fallback_count,
            "timeout_count": self.timeout_count,
            "hard_timeout_count": self.hard_timeout_count,
            "soft_stage_budget_exceeded": self.reason == "stage_budget_exceeded",
            "warnings": unique_strings(self.warnings),
        }


class NullStageRuntime:
    def __init__(self, name: str = "skipped") -> None:
        self.name = name
        self.status = "completed"

    def __enter__(self) -> "NullStageRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    @property
    def allowed(self) -> bool:
        return True

    def can_continue(self) -> bool:
        return True

    def mark_fallback(self, reason: str = "fallback") -> None:
        return None

    def mark_warning(self, warning: str) -> None:
        return None

    def record_call(self, duration_ms: float) -> None:
        return None

    def record_timeout(self, *, hard: bool = True) -> None:
        return None

    def record_fallback(self, reason: str = "fallback") -> None:
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "schema": "llm_stage_runtime.v1",
            "stage": self.name,
            "status": "skipped",
            "reason": "null_stage",
            "duration_ms": 0.0,
            "llm_duration_ms": 0.0,
            "call_count": 0,
            "avg_call_duration_ms": 0.0,
            "slow_call_count": 0,
            "fallback_count": 0,
            "timeout_count": 0,
            "hard_timeout_count": 0,
            "soft_stage_budget_exceeded": False,
            "warnings": [],
        }


def null_stage(name: str = "skipped") -> NullStageRuntime:
    return NullStageRuntime(name)
