import asyncio
from contextlib import AsyncExitStack
import threading
import time
from unittest.mock import Mock

import numpy as np
from openpi_client import msgpack_numpy
import pytest
from websockets.asyncio.client import connect

from openpi.serving.edf_scheduler import Reservation
from openpi.serving.edf_scheduler import can_reserve
from openpi.serving.edf_server import EDFConfig
from openpi.serving.edf_server import EDFRobotServer
from openpi.serving.edf_server import EDFSession
from openpi.serving.slot_scheduler import SchedulerConfig


def test_admission_checks_deadlines_and_nonpreemptive_blocking():
    def admissible(*reservations):
        return can_reserve(reservations, cost=0.06, max_utilization=0.85)

    assert admissible(Reservation(4, 0.15), Reservation(5, 0.2))
    # The short deadline is individually feasible, but cannot wait for a
    # non-preemptive job and then run within 100 ms.
    assert admissible(Reservation(3, 0.1))
    assert not admissible(Reservation(3, 0.1), Reservation(4, 0.2))
    assert not admissible(Reservation(3, 0.1), Reservation(4, 0.1))
    assert not admissible(Reservation(9, 0.2), Reservation(6, 0.2))


class VirtualPolicy:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_robot = None
        self.max_workers = 0
        self.running = 0
        self.calls = []
        self.delay = 0.045

    def infer(self, observation):
        robot = int(observation["state"][0])
        with self.lock:
            self.active_robot = robot
            self.running += 1
            self.max_workers = max(self.max_workers, self.running)
            self.calls.append(robot)
        try:
            time.sleep(self.delay)
            return {"actions": observation["state"][None, :] + 1}
        finally:
            with self.lock:
                self.running -= 1
                self.active_robot = None


async def run_server(*, prefetch=True, collect_time=0.022):
    server = EDFRobotServer(
        VirtualPolicy(),
        {"state": np.zeros(1, np.float32)},
        policy_id="virtual",
        input_spec={},
        action_horizon=30,
        host="127.0.0.1",
        port=0,
        config=SchedulerConfig(lead_time=0.018, heartbeat_timeout=2),
        edf=EDFConfig(prefetch=prefetch, max_collect_time=collect_time),
        warmup=0,
        runs=3,
    )
    try:
        await server.start()
    except BaseException:
        await server.close()
        raise
    return server


async def read(ws, message_type):
    async with asyncio.timeout(2):
        while True:
            message = msgpack_numpy.unpackb(await ws.recv())
            if message["type"] == "error":
                raise AssertionError(message)
            if message["type"] == message_type:
                return message


async def register(stack, server, robot, *, target=5, minimum=5, latency=200):
    ws = await stack.enter_async_context(connect(f"ws://127.0.0.1:{server.bound_port}"))
    await ws.send(
        msgpack_numpy.packb(
            {
                "type": "register",
                "robot_id": robot,
                "policy_id": "virtual",
                "input_spec": {},
                "target_hz": target,
                "min_hz": minimum,
                "action_hz": 30,
                "max_latency_ms": latency,
            }
        )
    )
    return ws, await read(ws, "register_ack")


async def observe(ws, request, robot):
    await ws.send(
        msgpack_numpy.packb(
            {
                "type": "observation",
                "session_id": request["session_id"],
                "request_id": request["request_id"],
                "observation": {"state": np.array([robot], np.float32)},
            }
        )
    )


@pytest.mark.parametrize("prefetch", [False, True])
def test_gpu_busy_prefetch_and_nonpreemptive_dispatch(prefetch):
    async def scenario():
        server = await run_server(prefetch=prefetch, collect_time=0.025)
        try:
            async with AsyncExitStack() as stack:
                first, ack = await register(stack, server, "first")
                assert ack["assigned_hz"] == 5
                # Leave enough time for the registration of the second robot
                # while the first inference is still in progress.
                server.policy.delay = 0.09
                await observe(first, await read(first, "collect"), 1)
                async with asyncio.timeout(1):
                    while server.policy.active_robot != 1:
                        await asyncio.sleep(0.001)
                other, ack = await register(stack, server, "second", target=2, minimum=2, latency=400)
                assert ack["assigned_hz"] == 2
                request = await read(other, "collect")
                if not prefetch:
                    assert server.policy.active_robot is None
                await observe(other, request, 2)
                assert (await read(first, "actions"))["result"]["actions"].item() == 2
                assert (await read(other, "actions"))["result"]["actions"].item() == 3
                assert server.policy.calls[-2:] == [1, 2]
                assert server.policy.max_workers == 1
                assert bool(server.prefetch_stats["requests"]) is prefetch
                assert bool(server.prefetch_stats["ready_while_busy"]) is prefetch
                assert bool(server.prefetch_stats["used"]) is prefetch
        finally:
            await server.close()

    asyncio.run(scenario())


def test_reduced_grant_and_dynamic_waitlist_keep_incumbent_rate():
    async def scenario():
        server = await run_server()
        try:
            async with AsyncExitStack() as stack:
                first, ack = await register(stack, server, "first", target=6, minimum=6, latency=300)
                assert ack["assigned_hz"] == 6
                incumbent = server.sessions["first"].reservation
                second, ack = await register(stack, server, "second", target=7, minimum=2, latency=300)
                assert 2 <= ack["assigned_hz"] < 7
                assert ack["slot_id"] is None
                waiting, ack = await register(stack, server, "waiting", target=5, minimum=2, latency=300)
                assert ack["status"] == "waitlisted"
                assert server.sessions["first"].reservation is incumbent
                old_generation = server.sessions["first"].generation
                await first.close()
                promoted = await read(waiting, "register_ack")
                assert promoted["status"] == "accepted"
                assert promoted["session_id"] != old_generation
                assert 2 <= promoted["assigned_hz"] <= 5
                assert server.sessions["second"].reservation is not None
                await observe(waiting, await read(waiting, "collect"), 3)
                assert (await read(waiting, "actions"))["result"]["actions"].item() == 4
                await second.close()
        finally:
            await server.close()

    asyncio.run(scenario())


def test_dispatch_chooses_earliest_ready_deadline_and_expires_stale_job():
    # Inject released jobs to test ordering without timing races between sockets.
    server = EDFRobotServer(Mock(), {}, policy_id="virtual", input_spec={})
    server._budget = 0.03  # noqa: SLF001
    server._dispatch = Mock()  # noqa: SLF001
    now = 10.0
    for robot, deadline, requested_at in (("late", 10.19, 9.99), ("first", 10.12, 9.99), ("stale", 10.09, 9.8)):
        server.sessions[robot] = EDFSession(
            robot,
            robot,
            Mock(),
            0.2,
            now,
            reservation=Reservation(5, 0.2),
            released=True,
            request=robot,
            next_release=10.2,
            deadline=deadline,
            requested_at=requested_at,
            observation={"state": robot},
        )
    try:
        server._step(now)  # noqa: SLF001
        server._dispatch.assert_called_once_with(server.sessions["first"], now)  # noqa: SLF001
        assert server.sessions["stale"].outgoing.get_nowait()["reason"] == "edf_deadline_or_observation_age"
        assert not server.sessions["stale"].released
    finally:
        server.executor.shutdown()


def test_collection_timeout_releases_worker_for_next_robot():
    async def scenario():
        server = await run_server(prefetch=False, collect_time=0.035)
        try:
            async with AsyncExitStack() as stack:
                silent, _ = await register(stack, server, "silent", target=3, minimum=3, latency=200)
                ignored = await read(silent, "collect")
                ready, _ = await register(stack, server, "ready", target=3, minimum=3, latency=300)
                dropped = await read(silent, "dropped")
                assert dropped["request_id"] == ignored["request_id"]
                assert dropped["reason"] == "edf_observation_timeout"
                collect = await read(ready, "collect")
                await observe(silent, ignored, 99)
                assert (await read(silent, "dropped"))["reason"] == "stale_or_duplicate_observation"
                await observe(ready, collect, 4)
                assert (await read(ready, "actions"))["result"]["actions"].item() == 5
                assert 99 not in server.policy.calls
        finally:
            await server.close()

    asyncio.run(scenario())
