"""Replay one observation against an existing server using independent robot sessions.

Measures delivered chunks and arrival gaps, not observation-to-action latency.
Never executes robot actions. See docs/multi_robot_experiment_zh.md.
"""

import argparse
import asyncio
from collections import Counter
import copy
import json
from pathlib import Path
import threading
import time

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.scheduled_client import run_robot


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--uri", default="ws://127.0.0.1:8000")
    result.add_argument("--sample", type=Path, required=True)
    result.add_argument("--policy-id", required=True)
    result.add_argument("--input-spec", default="{}")
    result.add_argument("--clients", type=int, default=1)
    result.add_argument("--robot-prefix", default="load")
    result.add_argument("--hz", type=float, default=5)
    result.add_argument("--min-hz", type=float, help="Minimum EDF reservation rate, defaults to --hz")
    result.add_argument("--action-hz", type=float, default=50)
    result.add_argument("--max-latency-ms", type=float, default=200)
    result.add_argument("--heartbeat-interval", type=float, default=0.5)
    result.add_argument("--warmup", type=float, default=5)
    result.add_argument("--duration", type=float, default=60)
    result.add_argument("--output", type=Path, required=True)
    return result


async def main(args):
    if args.clients < 1 or args.duration <= 0 or args.warmup < 0:
        raise ValueError("Require clients >= 1, duration > 0, warmup >= 0")
    sample = msgpack_numpy.unpackb(args.sample.read_bytes())
    spec = json.loads(args.input_spec)
    if not isinstance(sample, dict) or not isinstance(spec, dict):
        raise ValueError("sample and input-spec must be objects")
    started = time.monotonic()
    window_start = started + args.warmup
    window_end = window_start + args.duration
    lock = threading.Lock()
    records = {}
    tasks = []

    def callbacks(robot):
        record = {
            "status": "connecting",
            "ack": None,
            "events": Counter(),
            "arrivals": [],
            "shape": None,
            "last_problem": None,
        }
        records[robot] = record

        def actions(result):
            values = result["actions"]
            if not isinstance(values, np.ndarray) or values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError("Expected finite 2D actions array")
            now = time.monotonic()
            with lock:
                record["shape"] = list(values.shape)
                if window_start <= now < window_end:
                    record["arrivals"].append(now)

        def event(message):
            with lock:
                kind = message["type"]
                if kind == "register_ack":
                    record["ack"] = message
                    record["status"] = message["status"]
                    print(
                        f"{robot}: {message['status']} slot={message['slot_id']} assigned_hz={message['assigned_hz']}",
                        flush=True,
                    )
                elif kind in ("error", "suspended", "disconnected"):
                    record["status"] = kind
                    if kind != "disconnected":
                        record["last_problem"] = message
                        print(f"{robot}: {kind} reason={message.get('reason')}", flush=True)
                if window_start <= time.monotonic() < window_end:
                    reason = message.get("reason")
                    record["events"][kind if reason is None else f"{kind}:{reason}"] += message.get("count", 1)

        return actions, event

    try:
        for index in range(args.clients):
            robot = f"{args.robot_prefix}-{index}"
            on_actions, on_event = callbacks(robot)
            registration = {
                "robot_id": robot,
                "policy_id": args.policy_id,
                "input_spec": spec,
                "target_hz": args.hz,
                "min_hz": args.hz if args.min_hz is None else args.min_hz,
                "action_hz": args.action_hz,
                "max_latency_ms": args.max_latency_ms,
            }
            tasks.append(
                asyncio.create_task(
                    run_robot(
                        args.uri,
                        registration,
                        lambda: copy.deepcopy(sample),
                        on_actions,
                        on_event,
                        heartbeat_interval=args.heartbeat_interval,
                    )
                )
            )
        done, _ = await asyncio.wait(tasks, timeout=args.warmup + args.duration, return_when=asyncio.FIRST_COMPLETED)
        if done:
            for task in done:
                task.result()
            raise RuntimeError("A robot session ended before the measurement completed")
    finally:
        stopped = time.monotonic()
        # Snapshot before cleanup emits disconnected events.
        with lock:
            measured = max(0, min(stopped, window_end) - window_start)
            summary = {
                "uri": args.uri,
                "policy_id": args.policy_id,
                "clients": args.clients,
                "target_hz": args.hz,
                "requested_duration_s": args.duration,
                "measured_duration_s": measured,
                "warmup_s": args.warmup,
                "robots": {},
            }
            for robot, record in records.items():
                arrivals = record["arrivals"]
                gaps = np.diff(arrivals) * 1000
                summary["robots"][robot] = {
                    "status": record["status"],
                    "registration": record["ack"],
                    "action_chunks": len(arrivals),
                    "delivered_hz": len(arrivals) / measured if measured else 0,
                    "arrival_gap_p95_ms": float(np.percentile(gaps, 95)) if len(gaps) else None,
                    "arrival_gap_max_ms": float(np.max(gaps)) if len(gaps) else None,
                    "action_shape": record["shape"],
                    "last_problem": record["last_problem"],
                    "events": dict(record["events"]),
                }
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main(parser().parse_args()))
