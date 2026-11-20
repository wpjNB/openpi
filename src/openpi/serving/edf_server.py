"""Flexible periodic reservations, non-preemptive EDF, and optional JIT collection."""

import asyncio
from collections import deque
import contextlib
import dataclasses
import logging
import math
import time
import uuid

from openpi.serving.edf_scheduler import Reservation
from openpi.serving.edf_scheduler import can_reserve
from openpi.serving.multi_robot_server import MultiRobotServer
from openpi.serving.multi_robot_server import Session
from openpi.serving.multi_robot_server import validate_observation

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class EDFConfig:
    prefetch: bool = True
    max_collect_time: float = 0.03
    max_observation_age: float = 0.1

    def __post_init__(self):
        for value in (self.max_collect_time, self.max_observation_age):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("EDF collection and observation age budgets must be positive and finite")
        if self.max_observation_age < self.max_collect_time:
            raise ValueError("Observation age budget must cover the collection budget")


@dataclasses.dataclass
class EDFSession(Session):
    minimum_hz: float = 0
    target_hz: float = 0
    reservation: Reservation | None = None
    next_release: float = math.inf
    deadline: float = math.inf
    released: bool = False
    collect_estimate: float = 0
    collect_samples: deque[float] = dataclasses.field(default_factory=lambda: deque(maxlen=24))
    prefetched: bool = False


class EDFRobotServer(MultiRobotServer):
    """Uses the multi-robot wire protocol with a deadline per released reservation job."""

    def __init__(self, *args, edf: EDFConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.edf = edf or EDFConfig()
        self._wake = asyncio.Event()
        self._version = 0
        self._active_session = None
        self._active_started = 0.0
        self._inference_estimate = 0.0
        self._budget = 0.0
        self._prefetch_requests = 0
        self._prefetch_hits = 0
        self._prefetch_ready_while_busy = 0

    @property
    def prefetch_stats(self) -> dict[str, int]:
        return {
            "requests": self._prefetch_requests,
            "used": self._prefetch_hits,
            "ready_while_busy": self._prefetch_ready_while_busy,
        }

    def _configure_scheduler(self):
        self._inference_estimate = self.profile.latency_mean
        self._budget = self.profile.service_budget
        logger.info("Policy %s profile=%s scheduler=edf prefetch=%s", self.policy_id, self.profile, self.edf.prefetch)

    def _admissible(self, extra=None):
        tasks = [session.reservation for session in self.sessions.values() if session.reservation is not None]
        if extra is not None:
            tasks.append(extra)
        return can_reserve(
            tasks,
            cost=self._budget + self.edf.max_collect_time,
            max_utilization=self.config.max_utilization,
        )

    def _reserve(self, session):
        minimum = Reservation(session.minimum_hz, session.max_latency)
        if not self._admissible(minimum):
            return
        target = Reservation(session.target_hz, session.max_latency)
        if self._admissible(target):
            choice = target
        else:
            # Keep a feasible grant throughout the bounded search.
            choice = minimum
            lo, hi = minimum.hz, target.hz
            for _ in range(16):
                middle = (lo + hi) / 2
                candidate = Reservation(middle, session.max_latency)
                if self._admissible(candidate):
                    choice, lo = candidate, middle
                else:
                    hi = middle
        session.reservation = choice
        session.next_release = time.monotonic()
        self._version += 1
        self._wake.set()

    def _register(self, websocket, message):
        robot_id = message["robot_id"]
        if not isinstance(robot_id, str) or not 0 < len(robot_id) <= 128 or robot_id in self.sessions:
            raise ValueError("Invalid or already connected robot_id")
        if message["policy_id"] != self.policy_id or message["input_spec"] != self.input_spec:
            raise ValueError("Policy or input specification mismatch")
        target, minimum = float(message["target_hz"]), float(message["min_hz"])
        latency = float(message["max_latency_ms"]) / 1000
        action_hz = float(message["action_hz"])
        if not all(math.isfinite(x) and x > 0 for x in (target, minimum, latency, action_hz)) or minimum > target:
            raise ValueError("Invalid rates or latency")
        # Never grant a rate whose period exceeds the advertised action coverage.
        minimum = max(minimum, action_hz / self.action_horizon)
        if minimum > target:
            raise ValueError("Action horizon cannot cover one inference period")
        if not math.isfinite(1 / minimum):
            raise ValueError("Minimum rate implies an unrepresentable period")
        if not can_reserve(
            [Reservation(minimum, latency)],
            cost=self._budget + self.edf.max_collect_time,
            max_utilization=self.config.max_utilization,
        ):
            raise ValueError("Minimum reservation exceeds this deployment's service or latency budget")
        session = EDFSession(
            robot_id,
            uuid.uuid4().hex,
            websocket,
            latency,
            time.monotonic(),
            minimum_hz=minimum,
            target_hz=target,
            collect_estimate=min(self.config.lead_time or self.edf.max_collect_time, self.edf.max_collect_time),
        )
        self.sessions[robot_id] = session
        if not self.paused:
            self._reserve(session)
        return session

    def _ack(self, session):
        reservation = session.reservation
        self._send(
            session,
            {
                "type": "register_ack",
                "status": "accepted" if reservation else "waitlisted",
                "session_id": session.generation,
                "slot_id": None,
                "assigned_hz": reservation.hz if reservation else 0,
                "period_ms": reservation.period * 1000 if reservation else None,
                "schedule_version": self._version,
                "reason": "scheduler_suspended"
                if self.paused
                else (None if reservation else "insufficient_gpu_capacity"),
                "action_horizon": self.action_horizon,
                "profile": dataclasses.asdict(self.profile),
                "scheduler": "edf",
                "prefetch": self.edf.prefetch,
                "service_budget_ms": self._budget * 1000,
            },
        )

    def _remove(self, session):
        if not self._current(session):
            return
        del self.sessions[session.robot_id]
        self._clear_job(session)
        if session.reservation is not None:
            self._version += 1
            if not self.paused:
                for candidate in list(self.sessions.values()):
                    if candidate.reservation is None and self._current(candidate):
                        self._reserve(candidate)
                        if candidate.reservation is not None:
                            self._ack(candidate)
        self._wake.set()

    def _receive_observation(self, session, message):
        now = time.monotonic()
        if (
            session.request is None
            or message.get("request_id") != session.request
            or session.observation is not None
            or now > self._expiry(session)
        ):
            self._send(
                session,
                {
                    "type": "dropped",
                    "request_id": message.get("request_id"),
                    "reason": "stale_or_duplicate_observation",
                },
            )
            return
        validate_observation(message["observation"], self.sample_obs)
        session.last_seen = now
        session.observation = message["observation"]
        if session.prefetched and self._active_session is not None:
            self._prefetch_ready_while_busy += 1
        self._record_collect_time(session, now - session.requested_at)
        self._wake.set()

    def _record_collect_time(self, session, elapsed):
        session.collect_samples.append(elapsed)
        samples = sorted(session.collect_samples)
        p95 = samples[math.ceil(0.95 * len(samples)) - 1]
        session.collect_estimate = min(self.edf.max_collect_time, max(0.001, p95 * 1.2))

    def _expiry(self, session):
        latest = session.deadline - self._budget
        if session.request is None:
            return latest
        if session.observation is None:
            return min(latest, session.requested_at + self.edf.max_collect_time)
        return min(latest, session.requested_at + self.edf.max_observation_age)

    @staticmethod
    def _clear_job(session):
        session.released = False
        session.request = None
        session.observation = None
        session.prefetched = False

    def _drop_job(self, session, reason):
        request = session.request
        if reason == "edf_observation_timeout" and session.deadline - self._budget >= (
            session.requested_at + self.edf.max_collect_time
        ):
            self._record_collect_time(session, self.edf.max_collect_time)
        self._clear_job(session)
        self._send(session, {"type": "dropped", "request_id": request, "reason": reason})

    def _collect(self, session, now):
        session.prefetched = self._active_session is not None
        if session.prefetched:
            self._prefetch_requests += 1
        session.request = uuid.uuid4().hex
        session.requested_at = now
        self._send(
            session,
            {
                "type": "collect",
                "session_id": session.generation,
                "request_id": session.request,
                "schedule_version": self._version,
            },
        )

    async def _execute(self, session, request, observation, deadline):
        started = time.monotonic()
        try:
            await self._infer(session, request, observation, deadline)
        finally:
            self._inference_estimate = 0.75 * self._inference_estimate + 0.25 * (time.monotonic() - started)
            self._active_session = None
            self._wake.set()

    def _dispatch(self, session, now):
        request, observation, deadline = session.request, session.observation, session.deadline
        if session.prefetched:
            self._prefetch_hits += 1
        self._clear_job(session)
        self._active_session = session
        self._active_started = now
        self._inference = asyncio.create_task(self._execute(session, request, observation, deadline))

    def _step(self, now):
        wake_at = now + self.config.heartbeat_timeout / 4
        jobs = []
        for session in list(self.sessions.values()):
            if not self._current(session):
                continue
            if now - session.last_seen >= self.config.heartbeat_timeout:
                self._remove(session)
                self._close_peer(session, 1001, "Heartbeat timeout")
                continue
            if self.paused or session.reservation is None:
                continue
            if session.released and now >= self._expiry(session):
                self._drop_job(
                    session,
                    "edf_observation_timeout"
                    if session.request and session.observation is None
                    else "edf_deadline_or_observation_age",
                )
            if not self._current(session):
                continue
            if now >= session.next_release:
                period = session.reservation.period
                skipped = math.floor((now - session.next_release) / period)
                release = session.next_release + skipped * period
                session.next_release = release + period
                session.deadline = release + session.reservation.deadline
                session.released = True
                if skipped:
                    self._send(
                        session,
                        {"type": "dropped", "request_id": None, "reason": "edf_missed_releases", "count": skipped},
                    )
                if now >= self._expiry(session):
                    self._drop_job(session, "edf_deadline_or_observation_age")
            wake_at = min(wake_at, session.next_release)
            if self._current(session) and session.released:
                wake_at = min(wake_at, self._expiry(session))
                if session is not self._active_session:
                    jobs.append(session)
        if not jobs:
            return wake_at
        jobs.sort(key=lambda session: (session.deadline, session.robot_id))
        if self._active_session is not None:
            if self.edf.prefetch and not any(session.request for session in jobs):
                candidate = jobs[0]
                predicted_finish = max(now, self._active_started + self._inference_estimate)
                collect_at = predicted_finish - candidate.collect_estimate
                if predicted_finish + self._budget <= candidate.deadline:
                    if collect_at <= now:
                        self._collect(candidate, now)
                        wake_at = min(wake_at, self._expiry(candidate))
                    else:
                        wake_at = min(wake_at, collect_at)
            return wake_at

        earliest = jobs[0]
        ready = next((session for session in jobs if session.observation is not None), None)
        if ready is earliest or (
            ready is not None and now + 2 * self._budget + self.edf.max_collect_time <= earliest.deadline
        ):
            self._dispatch(ready, now)
            self._wake.set()
        elif earliest.request is None:
            if now + self.edf.max_collect_time + self._budget <= earliest.deadline:
                self._collect(earliest, now)
                wake_at = min(wake_at, self._expiry(earliest))
            else:
                self._drop_job(earliest, "edf_insufficient_collection_slack")
                self._wake.set()
        return wake_at

    async def _schedule(self):
        while True:
            self._wake.clear()
            wake_at = self._step(time.monotonic())
            if self._wake.is_set():
                continue
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(max(0, wake_at - time.monotonic())):
                    await self._wake.wait()

    def _overloaded(self, budget):
        # Never shrink the admitted budget based on a short run of fast calls.
        self._budget = max(self.profile.service_budget, budget)
        return not self._admissible()

    def _pause(self, reason):
        for session in self.sessions.values():
            self._clear_job(session)
        super()._pause(reason)
        self._wake.set()
