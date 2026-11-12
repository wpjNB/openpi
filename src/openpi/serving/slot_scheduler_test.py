# ruff: noqa: SLF001
import asyncio
import threading
import time

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.scheduled_client import run_robot
import pytest
from websockets.asyncio.client import connect

from openpi.serving.multi_robot_server import MultiRobotServer
from openpi.serving.multi_robot_server import validate_observation
from openpi.serving.slot_scheduler import PolicyProfile
from openpi.serving.slot_scheduler import SchedulerConfig
from openpi.serving.slot_scheduler import SlotPool
from openpi.serving.slot_scheduler import benchmark_policy


def test_capacity_stable_phases_and_reuse():
    pool = SlotPool(PolicyProfile(0.03, 0.03, 0.03, 33, 0.036), SchedulerConfig())
    assert len(pool.slots) == 4
    assert [pool.acquire(str(i)) for i in range(4)] == [0, 1, 2, 3]
    assert pool.acquire("overflow") is None
    pool.release("1")
    assert pool.acquire("replacement") == 1
    assert pool.slots == ["0", "replacement", "2", "3"]
    pool.release("1")  # Old generation cannot release the new owner's slot.
    assert pool.slots[1] == "replacement"
    with pytest.raises(ValueError, match="cannot satisfy"):
        SlotPool(PolicyProfile(0.2, 0.2, 0.2, 5, 0.2), SchedulerConfig())


@pytest.mark.parametrize(
    "kwargs",
    [{"target_hz": 0}, {"target_hz": float("nan")}, {"max_utilization": 1.1}, {"lead_time": 0.3}, {"online_window": 1}],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="Invalid|finite|lead_time"):
        SchedulerConfig(**kwargs)


def test_completed_benchmark():
    calls = []
    profile = benchmark_policy(lambda: calls.append(1), warmup=2, runs=4, synchronize=lambda: time.sleep(0.001))
    assert len(calls) == 6
    assert profile.latency_mean >= 0.001
    assert profile.service_budget >= profile.latency_p99
    assert profile.saturated_qps == 1 / profile.latency_mean


def test_profile_input_validation():
    sample = {"camera": np.zeros((2, 2, 3), np.uint8), "prompt": "long prompt"}
    validate_observation(dict(sample, prompt="short"), sample)
    validate_observation({"history": [sample]}, {"history": [sample]})
    with pytest.raises(ValueError, match="sequence"):
        validate_observation({"history": [sample, sample]}, {"history": [sample]})
    for observation in (
        dict(sample, camera=np.zeros((3, 2, 3), np.uint8)),
        dict(sample, prompt="x" * 50),
        {},
        dict(sample, camera=np.zeros((2, 2, 3))),
    ):
        with pytest.raises(ValueError, match="Observation"):
            validate_observation(observation, sample)


class FakePolicy:
    def __init__(self):
        self.delay = 0.012
        self.concurrent = 0
        self.max_concurrent = 0
        self.thread_ids = set()

    def infer(self, obs):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.thread_ids.add(threading.get_ident())
        try:
            time.sleep(self.delay)
            return {"actions": obs["state"][None, :] + 1}
        finally:
            self.concurrent -= 1


def registration(robot_id):
    return {
        "type": "register",
        "robot_id": robot_id,
        "policy_id": "fake",
        "input_spec": {"cameras": 0},
        "target_hz": 10,
        "min_hz": 5,
        "action_hz": 50,
        "max_latency_ms": 150,
    }


async def receive(ws, kind):
    async with asyncio.timeout(3):
        while True:
            message = msgpack_numpy.unpackb(await ws.recv())
            if message["type"] == kind:
                return message


async def make_server(**kwargs):
    server = MultiRobotServer(
        FakePolicy(),
        {"state": np.zeros(3, np.float32)},
        policy_id="fake",
        input_spec={"cameras": 0},
        host="127.0.0.1",
        port=0,
        config=SchedulerConfig(target_hz=10, lead_time=0.015, **kwargs),
        warmup=1,
        runs=3,
        margin=1.5,
    )
    await server.start()
    return server, f"ws://127.0.0.1:{server.bound_port}"


async def register(uri, robot_id):
    ws = await connect(uri, compression=None)
    await ws.send(msgpack_numpy.packb(registration(robot_id)))
    ack = await receive(ws, "register_ack")
    return ws, ack


def test_dynamic_registration_waitlist_and_generation_isolation():
    async def scenario():
        server, uri = await make_server()
        peers = []
        try:
            capacity = len(server.pool.slots)
            for i in range(capacity + 1):
                ws, ack = await register(uri, str(i))
                peers.append(ws)
                assert ack["status"] == ("accepted" if i < capacity else "waitlisted")
            assert server.sessions[str(capacity)].slot is None
            old_session = server.sessions["0"]
            old_slot = old_session.slot
            await peers[0].close()
            promoted = await receive(peers[-1], "register_ack")
            assert promoted["status"] == "accepted"
            assert promoted["slot_id"] == old_slot
            assert server.sessions[str(capacity)].generation != old_session.generation
            assert not server._current(old_session)
            duplicate = await connect(uri)
            await duplicate.send(msgpack_numpy.packb(registration("1")))
            assert (await receive(duplicate, "error"))["type"] == "error"
            await duplicate.close()
        finally:
            for ws in peers:
                await ws.close()
            await server.close()

    asyncio.run(scenario())


def test_multiple_clients_single_worker_and_client_callbacks():
    async def scenario():
        server, uri = await make_server()
        actions = {"a": [], "b": []}
        events = []
        clients = []
        try:
            for robot, results in actions.items():
                clients.append(
                    asyncio.create_task(
                        run_robot(
                            uri,
                            registration(robot),
                            lambda: {"state": np.array([1, 2, 3], np.float32)},
                            results.append,
                            events.append,
                            heartbeat_interval=0.05,
                        )
                    )
                )
            async with asyncio.timeout(4):
                while not all(len(results) >= 2 for results in actions.values()):
                    for task in clients:
                        if task.done():
                            task.result()
                    await asyncio.sleep(0.01)
            for results in actions.values():
                np.testing.assert_array_equal(results[0]["actions"], [[2, 3, 4]])
            assert server.policy.max_concurrent == 1
            assert len(server.policy.thread_ids) == 1  # Profiling and serving share the worker.
        finally:
            for task in clients:
                task.cancel()
            await asyncio.gather(*clients, return_exceptions=True)
            await server.close()
        assert sum(event["type"] == "disconnected" for event in events) == 2

    asyncio.run(scenario())


def test_heartbeat_expiry_promotes_waiter():
    async def scenario():
        server, uri = await make_server(heartbeat_timeout=0.3)
        peers = []
        try:
            for i in range(len(server.pool.slots)):
                ws, _ = await register(uri, str(i))
                peers.append(ws)
            await asyncio.sleep(0.15)
            waiter, ack = await register(uri, "waiter")
            peers.append(waiter)
            assert ack["status"] == "waitlisted"
            assert (await receive(waiter, "register_ack"))["status"] == "accepted"
        finally:
            for ws in peers:
                await ws.close()
            await server.close()

    asyncio.run(scenario())


def test_deadline_drop_and_sustained_overload_pause():
    async def scenario():
        server, uri = await make_server(online_window=2, overload_windows=2)
        ws, ack = await register(uri, "a")
        try:
            collect = await receive(ws, "collect")
            server.policy.delay = 0.18
            await ws.send(
                msgpack_numpy.packb(
                    {
                        "type": "observation",
                        "session_id": ack["session_id"],
                        "request_id": collect["request_id"],
                        "observation": {"state": np.zeros(3, np.float32)},
                    }
                )
            )
            # Ignore intervening collection notices: only the completed job may report a deadline violation.
            async with asyncio.timeout(3):
                while True:
                    dropped = await receive(ws, "dropped")
                    if dropped["request_id"] == collect["request_id"]:
                        break
            assert dropped["reason"] == "deadline_exceeded"
            # Monitor requires consecutive complete windows, not one outlier.
            server._latencies.clear()
            for _ in range(3):
                server._monitor(0.2)
            assert not server.paused
            server._monitor(0.2)
            assert server.paused
            assert (await receive(ws, "suspended"))["reason"] == "service_budget_exceeded"
        finally:
            await ws.close()
            await server.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    [
        {"target_hz": 5},
        {"input_spec": {}},
        {"policy_id": "wrong"},
        {"max_latency_ms": 1},
        {"min_hz": float("nan")},
        {"action_hz": 100000},
    ],
)
def test_incompatible_registration_rejected(change):
    async def scenario():
        server, uri = await make_server()
        try:
            async with connect(uri) as ws:
                await ws.send(msgpack_numpy.packb(dict(registration("bad"), **change)))
                assert (await receive(ws, "error"))["reason"]
            assert not server.sessions
        finally:
            await server.close()

    asyncio.run(scenario())


def test_disconnect_during_inference_never_delivers_to_reused_slot():
    async def scenario():
        server, uri = await make_server()
        old, ack = await register(uri, "robot")
        new = None
        try:
            request = await receive(old, "collect")
            server.policy.delay = 0.12
            await old.send(
                msgpack_numpy.packb(
                    {
                        "type": "observation",
                        "session_id": ack["session_id"],
                        "request_id": request["request_id"],
                        "observation": {"state": np.full(3, 999, np.float32)},
                    }
                )
            )
            async with asyncio.timeout(2):
                while not server.policy.concurrent:
                    await asyncio.sleep(0.001)
            await old.close()
            new, new_ack = await register(uri, "robot")
            assert new_ack["session_id"] != ack["session_id"]
            # A new connection receives only results for its own freshly captured observations.
            async with asyncio.timeout(2):
                while True:
                    message = msgpack_numpy.unpackb(await new.recv())
                    assert message["type"] != "actions"
                    if message["type"] == "collect":
                        if server.policy.concurrent:
                            continue
                        await new.send(
                            msgpack_numpy.packb(
                                {
                                    "type": "observation",
                                    "session_id": new_ack["session_id"],
                                    "request_id": message["request_id"],
                                    "observation": {"state": np.zeros(3, np.float32)},
                                }
                            )
                        )
                        result = await receive(new, "actions")
                        np.testing.assert_array_equal(result["result"]["actions"], [[1, 1, 1]])
                        break
            assert server.policy.max_concurrent == 1
        finally:
            await old.close()
            if new:
                await new.close()
            await server.close()

    asyncio.run(scenario())


def test_stale_observation_is_not_inferred():
    async def scenario():
        server, uri = await make_server()
        ws, ack = await register(uri, "a")
        try:
            await ws.send(
                msgpack_numpy.packb(
                    {
                        "type": "observation",
                        "session_id": ack["session_id"],
                        "request_id": "old",
                        "observation": {"state": np.zeros(3, np.float32)},
                    }
                )
            )
            assert (await receive(ws, "dropped"))["reason"] == "stale_or_duplicate_observation"
            assert server.policy.concurrent == 0
        finally:
            await ws.close()
            await server.close()

    asyncio.run(scenario())
