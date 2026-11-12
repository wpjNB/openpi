"""Capacity planning for a homogeneous, batch-one, non-preemptive policy worker."""

from collections.abc import Callable
import dataclasses
import math
import time

import numpy as np


@dataclasses.dataclass(frozen=True)
class PolicyProfile:
    latency_mean: float
    latency_p95: float
    latency_p99: float
    saturated_qps: float
    service_budget: float


def benchmark_policy(
    infer: Callable[[], object],
    *,
    warmup: int = 30,
    runs: int = 300,
    margin: float = 1.2,
    synchronize: Callable[[], None] = lambda: None,
) -> PolicyProfile:
    """Measure completed calls. infer must materialize outputs, or supply synchronize.

    saturated_qps is serial worker throughput, NOT multi-client/network throughput.
    Run on the same dedicated thread used for subsequent inference.
    """
    if warmup < 0 or runs < 2 or not math.isfinite(margin) or margin < 1:
        raise ValueError("Require warmup >= 0, runs >= 2, margin >= 1")
    for _ in range(warmup):
        infer()
        synchronize()
    latencies = []
    for _ in range(runs):
        synchronize()
        start = time.perf_counter()
        infer()
        synchronize()
        latencies.append(time.perf_counter() - start)
    mean = float(np.mean(latencies))
    p95, p99 = np.percentile(latencies, [95, 99])
    return PolicyProfile(mean, float(p95), float(p99), 1 / mean, float(p99) * margin)


@dataclasses.dataclass(frozen=True)
class SchedulerConfig:
    target_hz: float = 5.0
    max_utilization: float = 0.85
    lead_time: float = 0.015
    heartbeat_timeout: float = 3.0
    online_window: int = 100
    overload_windows: int = 3

    def __post_init__(self):
        for value in (self.target_hz, self.max_utilization, self.lead_time, self.heartbeat_timeout):
            if not math.isfinite(value):
                raise ValueError("Scheduler parameters must be finite")
        if self.target_hz <= 0 or not 0 < self.max_utilization <= 1 or self.heartbeat_timeout <= 0:
            raise ValueError("Invalid scheduler rate, utilization or heartbeat timeout")
        if not 0 <= self.lead_time < 1 / self.target_hz:
            raise ValueError("lead_time must be in [0, period)")
        if self.online_window < 2 or self.overload_windows < 1:
            raise ValueError("Invalid monitoring window")


class SlotPool:
    """Stable phases; owners are connection generation IDs, never reusable robot IDs."""

    def __init__(self, profile: PolicyProfile, config: SchedulerConfig):
        if not math.isfinite(profile.service_budget) or profile.service_budget <= 0:
            raise ValueError("Invalid service budget")
        self.period = 1 / config.target_hz
        capacity = math.floor(config.max_utilization * self.period / profile.service_budget)
        if capacity < 1:
            raise ValueError("Policy cannot satisfy the configured frequency and headroom")
        # One scheduler iteration collects then launches a slot. Reserve enough
        # phase spacing for the configured observation lead time as well as compute.
        if config.lead_time:
            capacity = min(capacity, math.floor(self.period / config.lead_time))
        self.slots: list[str | None] = [None] * capacity
        self.slot_duration = self.period / capacity
        self.version = 0

    def acquire(self, owner: str) -> int | None:
        if owner in self.slots:
            return self.slots.index(owner)
        for index, current in enumerate(self.slots):
            if current is None:
                self.slots[index] = owner
                self.version += 1
                return index
        return None

    def release(self, owner: str) -> None:
        if owner in self.slots:
            self.slots[self.slots.index(owner)] = None
            self.version += 1
