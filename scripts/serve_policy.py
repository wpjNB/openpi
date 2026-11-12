import asyncio
import dataclasses
import enum
import json
import logging
import pathlib
import socket

from openpi_client import msgpack_numpy
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import multi_robot_server
from openpi.serving import slot_scheduler
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Opt-in homogeneous, batch-one multi-robot scheduling. Legacy protocol remains the default.
    multi_robot: bool = False
    # Msgpack observation representative of the deployed input shapes and maximum text length.
    profile_observation: pathlib.Path | None = None
    # JSON object describing cameras/resolution/history; clients must match exactly.
    input_spec: str = "{}"
    # Stable slot capacity, observation lead time and online overload detection.
    scheduler: slot_scheduler.SchedulerConfig = dataclasses.field(default_factory=slot_scheduler.SchedulerConfig)
    # Number of completed warmup calls before measuring latency.
    profile_warmup: int = 30
    # Number of completed calls used for startup latency percentiles.
    profile_runs: int = 300
    # Multiply measured P99 by this factor for admission and monitoring.
    profile_margin: float = 1.2
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    if args.multi_robot:
        if args.profile_observation is None:
            raise ValueError("--multi-robot requires --profile-observation")
        sample = msgpack_numpy.unpackb(args.profile_observation.read_bytes())
        input_spec = json.loads(args.input_spec)
        if not isinstance(sample, dict) or not isinstance(input_spec, dict):
            raise ValueError("Profiling observation and input spec must be objects")
        checkpoint = args.policy if isinstance(args.policy, Checkpoint) else DEFAULT_CHECKPOINT[args.env]
        server = multi_robot_server.MultiRobotServer(
            policy,
            sample,
            policy_id=checkpoint.config,
            input_spec=input_spec,
            action_horizon=_config.get_config(checkpoint.config).model.action_horizon,
            config=args.scheduler,
            port=args.port,
            warmup=args.profile_warmup,
            runs=args.profile_runs,
            margin=args.profile_margin,
        )
        asyncio.run(server.run())
        return

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
