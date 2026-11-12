"""Async client for the opt-in multi-robot serving protocol (not legacy infer RPC)."""

import asyncio
import functools
import time

from openpi_client import msgpack_numpy

try:
    from websockets.asyncio.client import connect
except ImportError:  # websockets 11/12 on older robot Python runtimes.
    from websockets.legacy.client import connect


async def run_robot(uri, registration, observation_provider, on_actions, on_event, *, heartbeat_interval=0.5):
    """Run until disconnect/cancellation; callbacks execute off the network event loop.

    observation_provider() captures a fresh observation on each collect notice.
    on_actions(result) must enqueue actions without executing a blocking robot loop.
    on_event(message) must handle waitlisted, dropped and suspended states, including
    an application-specific safe stop on suspension/disconnect or buffer underflow.
    After connecting, on exit on_event receives disconnected, even after cancellation.
    No automatic reconnect, action replay, or control-rate negotiation is performed.
    """
    if heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be positive")
    loop = asyncio.get_running_loop()
    session_id = None
    capture = None
    heartbeat = None
    send_lock = asyncio.Lock()

    async with connect(uri, compression=None, max_size=16 * 1024 * 1024) as websocket:

        async def send(message):
            async with send_lock:
                await websocket.send(msgpack_numpy.packb(message))

        async def beat():
            while True:
                await asyncio.sleep(heartbeat_interval)
                if session_id:
                    await send(dict(type="heartbeat", session_id=session_id))

        async def collect(message):
            obs = await loop.run_in_executor(None, observation_provider)
            await send(
                dict(type="observation", session_id=session_id, request_id=message["request_id"], observation=obs)
            )

        try:
            await send(dict(registration, type="register"))
            heartbeat = asyncio.ensure_future(beat())
            pending = {}
            async for raw in websocket:
                message = msgpack_numpy.unpackb(raw)
                kind = message.get("type")
                if kind == "register_ack":
                    session_id = message["session_id"]
                elif kind == "collect":
                    # Keep at most one capture and one pending action request.
                    pending.clear()
                    pending[message["request_id"]] = time.monotonic()
                    if capture and capture.done():
                        capture.result()  # Propagate provider/send errors.
                    if capture is None or capture.done():
                        capture = asyncio.ensure_future(collect(message))
                    continue
                elif kind == "actions":
                    sent = pending.pop(message["request_id"], None)
                    if sent is not None and time.monotonic() - sent <= registration["max_latency_ms"] / 1000:
                        await loop.run_in_executor(None, functools.partial(on_actions, message["result"]))
                    else:
                        await loop.run_in_executor(
                            None, functools.partial(on_event, dict(type="dropped", reason="client_deadline"))
                        )
                    continue
                elif kind == "suspended":
                    pending.clear()
                await loop.run_in_executor(None, functools.partial(on_event, message))
        finally:
            for task in (capture, heartbeat):
                if task:
                    task.cancel()
            await asyncio.gather(*(task for task in (capture, heartbeat) if task), return_exceptions=True)
            await loop.run_in_executor(None, functools.partial(on_event, dict(type="disconnected")))
