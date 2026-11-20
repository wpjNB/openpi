# Multi-robot VLA serving (fixed-slot baseline)

On the EDF branch, `--multi-robot` defaults to EDF. See the [EDF and prefetch guide](multi_robot_edf_zh.md) for the new mode. The examples on this page explicitly select the fixed-slot baseline.

One loaded policy serves dynamically registered robots through one dedicated inference
thread, with batch size 1. Enable this protocol explicitly; the existing
`WebsocketPolicyServer` and `WebsocketClientPolicy` protocol remain the default.

## Try without a model or GPU

```bash
uv run --frozen examples/multi_robot/demo.py --clients 8 --duration 2
uv run --frozen pytest -q src/openpi/serving/slot_scheduler_test.py packages/openpi-client
```

The demo sleeps to simulate model work; its timings are **not VLA benchmarks**.
The server measures that work and derives capacity. `--clients` only controls the
load generator. Excess connections are waitlisted. Tests exercise real loopback
websockets, registration, promotion, heartbeat expiration, stale observations,
disconnection during inference, serial worker execution, deadlines and suspension.

## Start with a trained policy

Capture a representative raw robot observation (the same dictionary normally
passed to `WebsocketClientPolicy.infer`) in NumPy-aware msgpack:

```python
from pathlib import Path
from openpi_client import msgpack_numpy

Path("sample.msgpack").write_bytes(msgpack_numpy.packb(observation))
```

Use the actual camera shapes, dtypes, state structure, and maximum expected prompt
length. Strings longer than the sample and different keys/shapes/dtypes are rejected.
Do not profile only a short prompt if production tasks contain longer prompts.
The operator-supplied `input_spec` labels the deployment; actual observations are
also checked against the sample, not merely trusted based on this label.

```bash
uv run --frozen scripts/serve_policy.py \
  --multi-robot --scheduler-mode FIXED --profile-observation sample.msgpack \
  --input-spec '{"num_cameras":2,"history_frames":1,"image_size":[224,224]}' \
  --scheduler.target-hz 5 --scheduler.max-utilization 0.85 \
  --scheduler.lead-time 0.015 --profile-warmup 30 --profile-runs 300 \
  policy:checkpoint --policy.config pi05_droid --policy.dir /path/to/checkpoint
```

The observation/spec must match the selected checkpoint's input transforms; the
camera settings above are an example. Startup uses the same worker as serving,
warms the policy, and measures completed calls. The built-in openpi Policy converts
actions to host NumPy arrays before returning, completing JAX/PyTorch device work.
Custom policies must do likewise, or supply a synchronization callback to
`MultiRobotServer`; do not measure only asynchronous kernel submission.

The startup log and registration acknowledgments expose mean, P95, P99, a P99-based
service budget (default margin 1.2), and serial saturated worker QPS. This QPS does
**not** measure multi-client network throughput. Repeat profiling whenever model,
GPU, precision, input shape, denoising steps, horizon or backend changes. No CUDA
Graph implementation is added here; an existing graph-backed policy may be wrapped
if it meets the completed-output and request-independent-state contracts.

## Robot client

Use `openpi_client.scheduled_client.run_robot` with synchronous callbacks. This is a
server-driven streaming API, not an `infer()` RPC:

```python
import asyncio
from openpi_client.scheduled_client import run_robot

registration = {
    "robot_id": "robot_007",
    "policy_id": "pi05_droid",  # Training config name used by the server.
    "target_hz": 5.0,
    "min_hz": 3.0,
    "action_hz": 50.0,
    "max_latency_ms": 100.0,
    "input_spec": {"num_cameras": 2, "history_frames": 1, "image_size": [224, 224]},
}
asyncio.run(run_robot(
    "ws://server:8000", registration,
    observation_provider=capture_latest_observation,
    on_actions=enqueue_action_chunk,
    on_event=handle_scheduler_event,
))
```

The application provides these callbacks. Capture fresh observations on demand;
`enqueue_action_chunk` should quickly hand off to the robot's bounded action buffer.
`handle_scheduler_event` must implement the robot's agreed response to missing
updates, `suspended`, `disconnected` and action-buffer underflow. No robot motion,
action execution, automatic rate reduction, or universal safe-stop behavior is
implemented by this networking library. Callbacks run outside the network loop;
keep them bounded. Cancellation cannot forcibly stop a running capture callback.

The client sends heartbeats every 0.5 seconds by default. Configure this below the
server's heartbeat timeout (default 3 seconds). Waitlisted clients keep heartbeats
running and receive a new `register_ack` when a slot becomes available.

## Wire protocol and scheduling

All frames use the existing `msgpack_numpy` codec. The first client frame is
`{type: register, ...registration}`; the server returns `register_ack` containing
`status` (`accepted` or `waitlisted`), a unique `session_id`, `slot_id`,
`assigned_hz`, `period_ms`, `schedule_version`, `action_horizon` and `profile`.
The deployment's horizon must cover at least one inference period at `action_hz`.
Malformed/incompatible registration returns `error` and closes the connection.
Duplicate connected robot IDs are rejected, rather than replacing a live robot.

For accepted robots:

1. Server sends `collect` with `session_id`, `request_id`, `schedule_version`.
2. Client captures and sends `observation` with those IDs and `observation` data.
3. At the reserved slot the worker runs only if the observation arrived, no other
   inference is active, and the estimated completion fits the request deadline.
4. Server returns `actions` with IDs, `result` and `service_ms`, or `dropped` with
   a reason. Late/duplicate observations cannot replace a newer request.

`heartbeat` and `unregister` carry `session_id`. A reconnect creates a fresh ID;
old observations/results cannot cross into a new connection reusing the same slot
or robot ID. The server has no unbounded inference backlog: one staged observation
per session, one active inference, four outbound messages per connection. Slow
consumers are disconnected; inbound frames are limited to 16 MiB.

For period P, service budget C and utilization limit rho, compute capacity is
`floor(rho * P / C)`. The current sequential collect/launch loop additionally caps
capacity at `floor(P / lead_time)` when lead time is nonzero, ensuring each slot
has its configured observation window. These slots retain their phases regardless
of connection count. Disconnect/heartbeat expiration frees a slot and promotes
the oldest waiting compatible connection. Missed slots are skipped, never replayed
in a catch-up burst. Zero capacity fails startup; V1 does not negotiate a new rate.

The request deadline is measured on the server from issuing `collect`, including
observation lead time, through completed inference and outbound queue waiting.
The client also rejects results exceeding its local collect-to-receive limit.
These are soft deadlines: arbitrary network delays, client callback delays and
robot execution time cannot be bounded here. They are not a proof of end-to-end
real-time schedulability. Validate deployment network latency and action-buffer
coverage before robot use; P99 plus margin is not a worst-case guarantee.

Online monitoring evaluates P99-based budgets in successive non-overlapping
windows (100 samples by default). Three consecutive overloaded windows suspend all
scheduling and admission. Inference errors suspend immediately. Clients receive
`suspended`; existing frequencies are never silently changed. Suspension is latched:
resolve contention/reprofile by restarting the deployment; there is no automatic
resume that might oscillate or violate a robot's control requirements.

V1 supports one policy configuration and one common inference rate only. It does
not implement heterogeneous EDF scheduling, dynamic batching, multi-GPU workers,
per-robot model state or automatic frequency offers. GPU throughput and actual
checkpoint inference need validation on the target hardware. `/healthz` is a
liveness endpoint, including when scheduling is suspended. As with the legacy
server, authentication/TLS must be supplied by the deployment gateway if needed.

For a step-by-step Chinese experiment guide, including client-count sweeps and a
replay load generator, see [完整实验手册](multi_robot_experiment_zh.md).
