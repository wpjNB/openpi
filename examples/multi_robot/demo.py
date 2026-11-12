"""CPU-only demonstration; --clients controls simulated demand, not server capacity."""

import asyncio
import dataclasses
import time

import numpy as np
from openpi_client.scheduled_client import run_robot
import tyro

from openpi.serving.multi_robot_server import MultiRobotServer
from openpi.serving.slot_scheduler import SchedulerConfig


class DemoPolicy:
    def infer(self, observation):
        time.sleep(0.03)  # Simulated completed model work, not a measured VLA latency.
        return {"actions": np.zeros((30, 7), dtype=np.float32)}


@dataclasses.dataclass
class Args:
    clients: int = 8
    duration: float = 2.0


async def main(args: Args):
    if args.clients < 1 or args.duration <= 0:
        raise ValueError("clients and duration must be positive")
    server = MultiRobotServer(
        DemoPolicy(),
        {"state": np.zeros(7, np.float32)},
        policy_id="demo",
        input_spec={},
        action_horizon=30,
        config=SchedulerConfig(),
        host="127.0.0.1",
        port=0,
        warmup=2,
        runs=5,
    )
    clients = []
    counts = [0] * args.clients
    events = []
    try:
        await server.start()
        port = server.bound_port

        def record(index, result):
            counts[index] += 1

        for index in range(args.clients):
            clients.append(  # noqa: PERF401
                asyncio.create_task(
                    run_robot(
                        f"ws://127.0.0.1:{port}",
                        {
                            "robot_id": f"robot-{index}",
                            "policy_id": "demo",
                            "input_spec": {},
                            "target_hz": 5,
                            "min_hz": 3,
                            "action_hz": 50,
                            "max_latency_ms": 100,
                        },
                        lambda: {"state": np.zeros(7, np.float32)},
                        lambda result, index=index: record(index, result),
                        events.append,
                    )
                )
            )
        await asyncio.sleep(args.duration)
        for client in clients:
            if client.done():
                client.result()
        print(f"Measured service budget: {server.profile.service_budget * 1000:.1f} ms")
        print(f"Capacity: {len(server.pool.slots)}; connected: {len(server.sessions)}")
        print(f"Action chunks per client: {counts}")
    finally:
        for client in clients:
            client.cancel()
        await asyncio.gather(*clients, return_exceptions=True)
        await server.close()


if __name__ == "__main__":
    asyncio.run(main(tyro.cli(Args)))
