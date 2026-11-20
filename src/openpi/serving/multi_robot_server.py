"""Server-driven observation collection with a single serialized inference worker."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import functools
import logging
import math
import time
import uuid

import numpy as np
from openpi_client import msgpack_numpy
import websockets
from websockets.asyncio.server import serve

from openpi.serving.slot_scheduler import SchedulerConfig
from openpi.serving.slot_scheduler import SlotPool
from openpi.serving.slot_scheduler import benchmark_policy
from openpi.serving.websocket_policy_server import _health_check

logger = logging.getLogger(__name__)


def validate_observation(value, sample):
    """Enforce the profiled input structure, shapes, dtypes and text length bound."""
    if isinstance(sample, dict):
        if not isinstance(value, dict) or value.keys() != sample.keys():
            raise ValueError("Observation keys differ from profiling sample")
        for key in sample:
            validate_observation(value[key], sample[key])
    elif isinstance(sample, list | tuple):
        if type(value) is not type(sample) or len(value) != len(sample):
            raise ValueError("Observation sequence differs from profiling sample")
        for item, example in zip(value, sample, strict=True):
            validate_observation(item, example)
    elif isinstance(sample, np.ndarray):
        if not isinstance(value, np.ndarray) or value.shape != sample.shape or value.dtype != sample.dtype:
            raise ValueError("Observation shape/dtype differs from profiling sample")
    elif isinstance(sample, str):
        if not isinstance(value, str) or len(value) > len(sample):
            raise ValueError("Observation text exceeds profiled length")
    elif type(value) is not type(sample):
        raise ValueError("Observation type differs from profiling sample")


@dataclasses.dataclass
class Session:
    robot_id: str
    generation: str
    websocket: object
    max_latency: float
    last_seen: float
    outgoing: asyncio.Queue = dataclasses.field(default_factory=lambda: asyncio.Queue(maxsize=4))
    slot: int | None = None
    request: str | None = None
    requested_at: float = 0
    slot_time: float = 0
    observation: dict | None = None


class MultiRobotServer:
    """One model instance, one worker thread, batch=1, one observation per scheduled slot.

    Policy must have request-independent transforms/state (apart from RNG), and infer
    must return host-materialized actions. The built-in openpi Policy meets the latter
    contract. No per-robot reset is applied to the shared policy.
    """

    def __init__(
        self,
        policy,
        sample_obs: dict,
        *,
        policy_id: str,
        input_spec: dict,
        action_horizon: int = 50,
        config: SchedulerConfig | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        warmup: int = 30,
        runs: int = 300,
        margin: float = 1.2,
        synchronize=lambda: None,
    ):
        self.policy = policy
        self.sample_obs = sample_obs
        self.policy_id = policy_id
        self.input_spec = input_spec
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.action_horizon = action_horizon
        self.config = config or SchedulerConfig()
        self.host, self.port = host, port
        self.benchmark_options = {"warmup": warmup, "runs": runs, "margin": margin, "synchronize": synchronize}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-policy")
        self.sessions: dict[str, Session] = {}
        self.pool: SlotPool | None = None
        self.profile = None
        self.paused = False
        self._latencies = []
        self._overload_count = 0
        self._inference: asyncio.Task | None = None
        self._ticker: asyncio.Task | None = None
        self._server = None
        self._closing_tasks = set()

    async def start(self):
        loop = asyncio.get_running_loop()
        self.profile = await loop.run_in_executor(
            self.executor,
            functools.partial(benchmark_policy, lambda: self.policy.infer(self.sample_obs), **self.benchmark_options),
        )
        self._configure_scheduler()
        self._server = await serve(
            self._handle,
            self.host,
            self.port,
            max_size=16 * 1024 * 1024,
            compression=None,
            process_request=_health_check,
        )
        self._ticker = asyncio.create_task(self._schedule())
        return self

    def _configure_scheduler(self):
        self.pool = SlotPool(self.profile, self.config)
        logger.info("Policy %s profile=%s capacity=%d", self.policy_id, self.profile, len(self.pool.slots))

    @property
    def bound_port(self) -> int:
        if self._server is None:
            raise RuntimeError("Server has not started")
        return self._server.sockets[0].getsockname()[1]

    async def close(self):
        if self._ticker:
            self._ticker.cancel()
            await asyncio.gather(self._ticker, return_exceptions=True)
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        # Never cancel an executor future: the underlying GPU work cannot be preempted.
        if self._inference:
            await self._inference
        if self._closing_tasks:
            await asyncio.gather(*self._closing_tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)

    async def run(self):
        try:
            await self.start()
            await self._ticker
        finally:
            await self.close()

    def _close_peer(self, session, code, reason):
        task = asyncio.create_task(session.websocket.close(code=code, reason=reason))
        self._closing_tasks.add(task)
        task.add_done_callback(self._closing_tasks.discard)

    def _send(self, session, message):
        try:
            session.outgoing.put_nowait(message)
        except asyncio.QueueFull:
            # Retire immediately; closing a stalled network peer must not block scheduling.
            self._remove(session)
            self._close_peer(session, 1013, "Slow consumer")

    async def _writer(self, session):
        while True:
            message = await session.outgoing.get()
            deadline = message.pop("_deadline", None)
            if deadline is not None and time.monotonic() > deadline:
                message = {"type": "dropped", "request_id": message["request_id"], "reason": "delivery_deadline"}
            await session.websocket.send(msgpack_numpy.packb(message))

    def _current(self, session):
        return self.sessions.get(session.robot_id) is session

    def _ack(self, session):
        self._send(
            session,
            {
                "type": "register_ack",
                "status": "accepted" if session.slot is not None else "waitlisted",
                "session_id": session.generation,
                "slot_id": session.slot,
                "assigned_hz": self.config.target_hz if session.slot is not None else 0,
                "period_ms": self.pool.period * 1000,
                "schedule_version": self.pool.version,
                "reason": "scheduler_suspended"
                if self.paused
                else (None if session.slot is not None else "insufficient_gpu_capacity"),
                "action_horizon": self.action_horizon,
                "profile": dataclasses.asdict(self.profile),
            },
        )

    def _promote(self):
        if self.paused:
            return
        for session in list(self.sessions.values()):
            if session.slot is None:
                session.slot = self.pool.acquire(session.generation)
                if session.slot is None:
                    break
                self._ack(session)

    def _remove(self, session):
        if self._current(session):
            del self.sessions[session.robot_id]
            self.pool.release(session.generation)
            session.request = None
            session.observation = None
            self._promote()

    def _register(self, websocket, message):
        robot_id = message["robot_id"]
        if not isinstance(robot_id, str) or not 0 < len(robot_id) <= 128 or robot_id in self.sessions:
            raise ValueError("Invalid or already connected robot_id")
        if message["policy_id"] != self.policy_id or message["input_spec"] != self.input_spec:
            raise ValueError("Policy or input specification mismatch")
        target, minimum = float(message["target_hz"]), float(message["min_hz"])
        latency = float(message["max_latency_ms"]) / 1000
        action_hz = float(message["action_hz"])
        if (
            not math.isfinite(action_hz)
            or action_hz <= 0
            or target <= 0
            or self.action_horizon / action_hz < 1 / target
        ):
            raise ValueError("Action horizon cannot cover one inference period")
        if not all(math.isfinite(x) for x in (target, minimum, latency)):
            raise ValueError("Non-finite robot requirements")
        if target != self.config.target_hz or not 0 < minimum <= target:
            raise ValueError("V1 requires the deployment's common target_hz")
        if latency < self.config.lead_time + self.profile.service_budget:
            raise ValueError("Latency budget cannot cover observation lead time and inference")
        session = Session(robot_id, uuid.uuid4().hex, websocket, latency, time.monotonic())
        self.sessions[robot_id] = session
        if not self.paused:
            session.slot = self.pool.acquire(session.generation)
        return session

    async def _handle(self, websocket):
        session = None
        writer = None
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=self.config.heartbeat_timeout)
            message = msgpack_numpy.unpackb(raw)
            if message.get("type") != "register":
                raise ValueError("First message must be register")
            session = self._register(websocket, message)
            writer = asyncio.create_task(self._writer(session))
            self._ack(session)
            if self.paused:
                self._send(
                    session, {"type": "suspended", "reason": "scheduler_suspended", "session_id": session.generation}
                )
            async for raw in websocket:
                message = msgpack_numpy.unpackb(raw)
                if not self._current(session):
                    break
                if message.get("session_id") != session.generation:
                    raise ValueError("Session mismatch")
                kind = message.get("type")
                if kind == "heartbeat":
                    session.last_seen = time.monotonic()
                elif kind == "observation":
                    self._receive_observation(session, message)
                elif kind == "unregister":
                    break
                else:
                    raise ValueError("Unknown message type")
        except websockets.ConnectionClosed:
            pass
        except (TimeoutError, ValueError, KeyError, TypeError, AttributeError) as exc:
            # Protocol errors are confined to this connection.
            await websocket.send(msgpack_numpy.packb({"type": "error", "reason": str(exc)}))
        finally:
            if session:
                self._remove(session)
            if writer:
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)

    def _receive_observation(self, session, message):
        if (
            session.request is None
            or message.get("request_id") != session.request
            or time.monotonic() > session.slot_time
            or session.observation is not None
        ):
            self._send(session, {"type": "dropped", "reason": "stale_or_duplicate_observation"})
            return
        validate_observation(message["observation"], self.sample_obs)
        session.last_seen = time.monotonic()
        session.observation = message["observation"]

    async def _sleep_until(self, when):
        await asyncio.sleep(max(0, when - time.monotonic()))

    async def _schedule(self):
        epoch = time.monotonic() + self.config.lead_time + 0.05
        index = 0
        while True:
            slot_time = epoch + index * self.pool.slot_duration
            await self._sleep_until(slot_time - self.config.lead_time)
            now = time.monotonic()
            for session in list(self.sessions.values()):
                if now - session.last_seen > self.config.heartbeat_timeout:
                    self._remove(session)
                    self._close_peer(session, 1001, "Heartbeat timeout")
            owner = self.pool.slots[index % len(self.pool.slots)]
            session = next((s for s in self.sessions.values() if s.generation == owner), None)
            if session and not self.paused and now < slot_time:
                session.request = uuid.uuid4().hex
                session.requested_at = now
                session.slot_time = slot_time
                session.observation = None
                self._send(
                    session,
                    {
                        "type": "collect",
                        "session_id": session.generation,
                        "request_id": session.request,
                        "schedule_version": self.pool.version,
                    },
                )
            await self._sleep_until(slot_time)
            if session and self._current(session) and not self.paused and session.request is not None:
                request, observation = session.request, session.observation
                session.request = None
                session.observation = None
                deadline = session.requested_at + session.max_latency
                busy = self._inference is not None and not self._inference.done()
                if observation is None or busy or time.monotonic() + self.profile.service_budget > deadline:
                    self._send(
                        session,
                        {
                            "type": "dropped",
                            "request_id": request,
                            "reason": "missing_observation_or_worker_busy_or_deadline",
                        },
                    )
                else:
                    self._inference = asyncio.create_task(self._infer(session, request, observation, deadline))
            index += 1
            # Skip elapsed slots instead of producing a catch-up burst.
            index = max(index, math.ceil((time.monotonic() - epoch) / self.pool.slot_duration))

    async def _infer(self, session, request, observation, deadline):
        start = time.monotonic()
        try:
            result = await asyncio.get_running_loop().run_in_executor(self.executor, self._infer_completed, observation)
            elapsed = time.monotonic() - start
            self._monitor(elapsed)
            if self._current(session) and not self.paused:
                if time.monotonic() <= deadline:
                    self._send(
                        session,
                        {
                            "type": "actions",
                            "session_id": session.generation,
                            "request_id": request,
                            "result": result,
                            "service_ms": elapsed * 1000,
                            "_deadline": deadline,
                        },
                    )
                else:
                    self._send(session, {"type": "dropped", "request_id": request, "reason": "deadline_exceeded"})
        except Exception:
            logger.exception("Policy inference failed; suspending scheduling")
            self._pause("policy_error")

    def _infer_completed(self, observation):
        result = self.policy.infer(observation)
        self.benchmark_options["synchronize"]()
        return result

    def _monitor(self, elapsed):
        self._latencies.append(elapsed)
        if len(self._latencies) < self.config.online_window:
            return
        budget = float(np.percentile(self._latencies, 99)) * self.benchmark_options["margin"]
        self._latencies.clear()
        overloaded = self._overloaded(budget)
        self._overload_count = self._overload_count + 1 if overloaded else 0
        if self._overload_count >= self.config.overload_windows:
            self._pause("service_budget_exceeded")

    def _overloaded(self, budget):
        active = sum(s.slot is not None for s in self.sessions.values())
        return active * self.config.target_hz * budget > self.config.max_utilization or budget > self.pool.slot_duration

    def _pause(self, reason):
        self.paused = True
        for session in list(self.sessions.values()):
            session.request = None
            session.observation = None
            self._send(session, {"type": "suspended", "reason": reason, "session_id": session.generation})
