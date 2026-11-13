# 多客户端 VLA 调度实验：从零跑通到 GPU 容量测试

这份手册验证的是：**一个服务端只加载一份 Policy，多个客户端动态连接，服务端按实测能力分配推理时隙**。客户端模拟机器人发送观测并接收动作，不连接机械臂，也不执行动作。

建议先完成 CPU 实验，再做 GPU 实验。CPU 实验不需要权重；GPU 实验需要完整 checkpoint 和相应显存。当前实现为 Batch=1、一个推理 Worker、统一推理频率，不做动态合批。

配套文件：

- `examples/multi_robot/demo.py`：一个命令启动模拟 Policy 和多个客户端。
- `examples/multi_robot/load_test.py`：连接独立服务端，重放观测，输出逐客户端 JSON 指标。
- `docs/multi_robot_serving.md`：协议及实现边界。

本手册和 `load_test.py` 是在初始调度提交 `ab77d20` 之后新增的文件。在其他机器运行前，先同步包含这些文件的版本；只有 `ab77d20` 的旧检出还没有负载脚本。所有命令均从 **openpi 仓库根目录**运行。

## 1. 到底需要模拟几个客户端？

先用 **1 个**验证单客户端推理，再测 **2 个**验证共享同一模型。随后按服务端测得的容量 N 测试 `N−1、N、N+1、2N`，去掉重复值和小于 1 的值。

| 阶段 | 客户端数量 | 验证目标 |
| --- | --- | --- |
| CPU 快速演示 | 1、2、4、8 | 正常服务、共享模型、容量边界、等待队列 |
| GPU 单客户端 | 1 | 权重、输入、GPU、推理链路均可用 |
| GPU 并发基线 | 2 | 两个客户端轮流获得动作，推理保持串行 |
| GPU 容量扫描 | N−1、N、N+1、2N | 找到准入边界，并观察过载连接等待 |
| 动态上下线 | N 个占位者 + 1 个等待者 | 断开一台后，等待者自动晋升 |

例如测得 N=4，就测 `1、2、3、4、5、8`。**8 个客户端不代表同时执行 8 次推理**；容量为 4 时，其余 4 个连接应该等待。不要把 5 台或 8 台写死进 Server。

## 2. 准备代码和 Python 环境

### 已配置好的云环境

```bash
cd /workspace/openpi
git branch --show-current
# 应为 feat/multi-client-vla-scheduler

test -f examples/multi_robot/load_test.py
.venv/bin/python --version
source .venv/bin/activate
```

后面所有 `python` 都使用这个虚拟环境。新开终端也需要在仓库根目录执行 `source .venv/bin/activate`。

### 自己的 Linux / GPU 机器

如果还没有仓库，先克隆；已有仓库则在保存自己的改动后切换到实验分支。不要对有本地修改的目录执行重置。

```bash
git clone --branch feat/multi-client-vla-scheduler --recurse-submodules https://github.com/wpjNB/openpi.git
cd openpi
```

确认已经同步配套 `load_test.py`。安装 uv 的方法见 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。然后执行：

```bash
export GIT_LFS_SKIP_SMUDGE=1
export UV_SYSTEM_CERTS=true
git submodule update --init --recursive
uv sync --frozen --all-extras --dev
source .venv/bin/activate

# README 要求的 PyTorch transformers 补丁，仅修改虚拟环境。
python - <<'PY'
import pathlib
import shutil
import transformers
source = pathlib.Path('src/openpi/models_pytorch/transformers_replace')
target = pathlib.Path(transformers.__file__).parent
shutil.copytree(source, target, dirs_exist_ok=True)
print('Patched:', target)
PY
```

Python 版本由仓库固定为 3.11。系统需要 FFmpeg 和 OpenGL 运行库，细节参见 README 和 CI；若 `cv2` 导入报告缺少 `libGL.so.1`，先安装本机发行版的对应系统库。

## 3. CPU 快速实验：先确认调度系统能跑

```bash
python examples/multi_robot/demo.py --clients 1 --duration 10
python examples/multi_robot/demo.py --clients 2 --duration 10
python examples/multi_robot/demo.py --clients 4 --duration 10
python examples/multi_robot/demo.py --clients 8 --duration 10
```

每条命令都启动自己的临时服务端和客户端，实验结束自动关闭，不必额外开服务端。

示例输出如下，实际数字由当前机器测量决定：

```text
Measured service budget: 36.1 ms
Capacity: 4; connected: 8
Action chunks per client: [50, 50, 50, 49, 0, 0, 0, 0]
```

解释：

- `connected=8`：8 个客户端连接成功。
- `Capacity=4`：只有 4 个获得推理时隙。
- 获准客户端约每秒收到 5 个动作块；等待者收到 0 个是预期行为。
- 计数存在边界差异，例如 49 与 50，不要求完全相等。
- 模拟 Policy 只是休眠约 30 ms，**不是实际 VLA 模型性能，也不是 GPU 基准**。

若机器负载较高，容量可能小于 4；应以打印值为准。

再运行现有回归测试：

```bash
JAX_PLATFORMS=cpu python -m pytest -q src/openpi/serving/slot_scheduler_test.py packages/openpi-client
```

测试覆盖动态注册、等待晋升、心跳超时、旧会话隔离、双客户端单 Worker、超时丢弃和过载暂停。该测试集在编写手册时执行了 41 项测试；以实际输出为准。

**CPU 实验完成条件：**有获准客户端持续收到动作，超容量客户端保持等待，测试通过。CPU 演示不验证真实模型、真实网络或机械臂动作质量。

## 4. GPU 实验：检查设备并准备 checkpoint

以下以仓库自带 `pi05_droid` 配置为例。它在当前配置中的 Action Horizon 为 **15**，不是之前假设的 30。输入为两路 224×224 图像、7 维关节位置、1 维夹爪位置和 prompt。

在 GPU 机器上执行：

```bash
unset JAX_PLATFORMS
nvidia-smi
python - <<'PY'
import jax
from openpi.training import config
print('Devices:', jax.devices())
print('Action horizon:', config.get_config('pi05_droid').model.action_horizon)
assert any(device.platform == 'gpu' for device in jax.devices()), 'JAX 未使用 GPU'
PY
```

如果使用本手册的 JAX checkpoint，检查 JAX GPU，而不只检查 `torch.cuda.is_available()`。README 给出的推理显存参考为 >8 GB，实际还受模型、输入和软件版本影响。

为实验输出创建目录，仓库默认忽略 `data/`：

```bash
mkdir -p data/multi_robot_experiment/checkpoints
export OPENPI_DATA_HOME="$PWD/data/multi_robot_experiment/cache"
mkdir -p "$OPENPI_DATA_HOME"
git rev-parse HEAD > data/multi_robot_experiment/code_commit.txt
nvidia-smi > data/multi_robot_experiment/gpu.txt
```

已有本地 `pi05_droid` checkpoint 可直接使用。没有的话，可以用已安装的 Google Cloud SDK / `gsutil` 下载公开权重：

```bash
# 需要先安装 gsutil；安装方法见 Google Cloud 官方文档。
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid data/multi_robot_experiment/checkpoints/
```

完整 checkpoint 应包含 `params/` 和 `assets/`。下载体积较大，不要为每个客户端重复下载。也不要使用只有模型参数、没有归一化统计的残缺目录。

```bash
export CHECKPOINT_DIR="$PWD/data/multi_robot_experiment/checkpoints/pi05_droid"
test -d "$CHECKPOINT_DIR/params"
test -d "$CHECKPOINT_DIR/assets"
```

如果权重在其他位置，只需修改 `CHECKPOINT_DIR`。模型初始化还可能下载公开分词器，需允许访问 Google Cloud Storage。可按照仓库的缓存路径提前获取：

```bash
mkdir -p "$OPENPI_DATA_HOME/big_vision"
curl --fail --location --output "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
```

下载失败时检查网络和错误响应，不要使用空文件代替权重或分词器，也不要关闭 TLS 验证。

## 5. 生成可重复的观测样本

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from openpi_client import msgpack_numpy

rng = np.random.default_rng(42)
observation = {
    'observation/exterior_image_1_left': rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    'observation/wrist_image_left': rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    'observation/joint_position': np.zeros(7, dtype=np.float64),
    'observation/gripper_position': np.zeros(1, dtype=np.float64),
    'prompt': 'pick up the object',
}
path = Path('data/multi_robot_experiment/droid_sample.msgpack')
path.write_bytes(msgpack_numpy.packb(observation))
print(path)
PY
```

所有模拟客户端重放这份样本，不涉及摄像头。随机图像和零状态只能用于检查协议、形状和吞吐，不能证明机器人任务成功。评估真实部署时，再用有代表性的录制数据替换样本，保持输入键、shape、dtype 一致，并重新 profiling。

服务端对 prompt 长度设有样本上界。改变任务文本、历史帧或图像规格时，使用覆盖实际最坏输入的样本重新测试；字符长度上界不等价于完整的 token 成本保证。

## 6. 终端 A：启动真实 Policy 服务端

在激活虚拟环境、设置好上述 `CHECKPOINT_DIR` 和 `OPENPI_DATA_HOME` 的终端中：

```bash
set -o pipefail
python -u scripts/serve_policy.py \
  --multi-robot \
  --port 8000 \
  --profile-observation data/multi_robot_experiment/droid_sample.msgpack \
  --input-spec '{"num_cameras":2,"history_frames":1,"image_size":[224,224]}' \
  --scheduler.target-hz 5 \
  --scheduler.max-utilization 0.85 \
  --scheduler.lead-time 0.015 \
  --scheduler.heartbeat-timeout 3 \
  --profile-warmup 30 \
  --profile-runs 300 \
  --profile-margin 1.2 \
  policy:checkpoint --policy.config pi05_droid --policy.dir "$CHECKPOINT_DIR" \
  2>&1 | tee data/multi_robot_experiment/server.log
```

先加载一次模型，再预热和测量；这段时间客户端连接失败是正常的。等待日志出现 `profile=... capacity=N`，然后在另一个终端验证：

```bash
curl --fail http://127.0.0.1:8000/healthz
```

应返回 `OK`。`/healthz` 只是进程存活检查，不能单独证明模型推理正常。

本次容量计算为：

```text
C = 实测 P99 × 1.2
P = 1 / 5 = 0.2 秒
N_compute = floor(0.85 × P / C)
N_lead = floor(P / 0.015)
N = min(N_compute, N_lead)
```

所有延迟字段以秒为单位。日志中的 `saturated_qps` 是串行 Worker 的吞吐估计，不是客户端实测总吞吐。**客户端数量应围绕测得的 N 扫描，而不是从“36 QPS”直接认定能接 7 台。**

如果启动报告无法满足目标频率，说明当前配置的 N=0。此时不要绕过准入控制。可以重新设计一个较低频率的纯性能实验，但须同时检查动作覆盖长度：`action_horizon / action_hz >= 1 / target_hz`。例如 horizon=15、action_hz=50 只能覆盖 0.3 秒，不能直接改成 2 Hz（周期 0.5 秒）。真实机器人需要重新设计动作缓冲或部署配置。

本手册使用仓库现有推理后端，没有自动启用 CUDA Graph，也没有自动把 horizon 改成 30。对比不同优化方案时要明确记录这些差异。

## 7. 终端 B：单客户端基线

```bash
# 先进入同一份 openpi 仓库根目录。
source .venv/bin/activate

LOAD_ARGS=(
  --uri ws://127.0.0.1:8000
  --sample data/multi_robot_experiment/droid_sample.msgpack
  --policy-id pi05_droid
  --input-spec '{"num_cameras":2,"history_frames":1,"image_size":[224,224]}'
  --hz 5
  --action-hz 50
  --max-latency-ms 200
  --warmup 5
  --duration 60
)

python examples/multi_robot/load_test.py "${LOAD_ARGS[@]}" \
  --clients 1 --output data/multi_robot_experiment/clients_1.json
```

正常应先看到：

```text
load-0: accepted slot=0 assigned_hz=5.0
```

5 秒热身结束后统计 60 秒，输出 JSON 包含：

- 每个客户端的注册状态、时隙和服务端 profile。
- `action_chunks` 和 `delivered_hz`：测量窗口内收到的有效动作块及频率。
- `arrival_gap_p95_ms` / `arrival_gap_max_ms`：相邻动作块**到达间隔**。
- `events`：窗口内丢弃、暂停等事件的原因计数。
- `action_shape`：动作块形状；当前 DROID 示例预期为 `[15, 8]`。

在稳定且未丢请求的回环实验中，5 Hz 对应动作到达间隔约 200 ms，60 秒内约 300 个动作块。不要要求边界计数恰好等于 300。

**到达间隔不是端到端请求延迟。**本脚本没有输出采集通知→动作接收的 P99，也没有记录机器人执行延迟。不能把 200 ms 到达间隔写成“模型推理耗时 200 ms”。

如果单客户端没有动作、发生 `suspended` 或大量 `dropped`，先排查单客户端，不要继续增加负载。

## 8. 自动生成并执行客户端数量扫描

先单独跑 2 个客户端：

```bash
python examples/multi_robot/load_test.py "${LOAD_ARGS[@]}" \
  --clients 2 --output data/multi_robot_experiment/clients_2.json
```

然后从单客户端 JSON 的 profile 自动算出扫描数量，保持和本手册的服务端参数一致：

```bash
python - <<'PY'
import json
import math
from pathlib import Path

root = Path('data/multi_robot_experiment')
result = json.loads((root / 'clients_1.json').read_text())
robot = next(iter(result['robots'].values()))
assert robot['registration']['status'] == 'accepted'
assert robot['action_chunks'] > 0, '单客户端尚未跑通'
c = robot['registration']['profile']['service_budget']
hz, rho, lead = 5.0, 0.85, 0.015
n = min(math.floor(rho / (hz * c)), math.floor(1 / (hz * lead)))
assert n >= 1
counts = sorted({1, 2, max(1, n - 1), n, n + 1, 2 * n})
(root / 'sweep_counts.txt').write_text(''.join(f'{count}\n' for count in counts))
print('Capacity:', n, 'Client counts:', counts)
PY

while read -r clients; do
  python examples/multi_robot/load_test.py "${LOAD_ARGS[@]}" \
    --clients "$clients" \
    --output "data/multi_robot_experiment/sweep_${clients}.json" || break
done < data/multi_robot_experiment/sweep_counts.txt
```

这里每轮运行结束会关闭全部模拟客户端，再开始下一轮；服务端及模型持续运行。每轮约 65 秒。如果出现过载暂停，应先结束扫描、排查并重启服务端；暂停状态不会自动恢复。

预期现象：

| 连接数量 M | 预期 |
| --- | --- |
| M ≤ N | 所有兼容客户端 accepted，各自接近目标频率 |
| M > N | N 个 accepted，其他 waitlisted；等待者不应收到动作 |
| M 继续增加 | 已准入客户端频率不应因无限排队而持续下降 |

达到 N 不等于通过稳定性测试；若动作频率明显不足或持续丢弃，即使容量公式允许，也需要排查网络、观测准备和预算估计。

## 9. 动态加入、退出和等待晋升

保持终端 A 的服务端运行。以下用 N=4 举例，**将占位者数量替换为当前测得的 N**。

终端 B：占满所有时隙，运行 30 秒（加 1 秒热身）：

```bash
python examples/multi_robot/load_test.py "${LOAD_ARGS[@]}" \
  --clients 4 --robot-prefix holder --warmup 1 --duration 30 \
  --output data/multi_robot_experiment/holders.json
```

看到全部 `holder-* accepted` 后，立即在终端 C 激活环境，运行等待者：

```bash
source .venv/bin/activate
python examples/multi_robot/load_test.py \
  --uri ws://127.0.0.1:8000 \
  --sample data/multi_robot_experiment/droid_sample.msgpack \
  --policy-id pi05_droid \
  --input-spec '{"num_cameras":2,"history_frames":1,"image_size":[224,224]}' \
  --hz 5 --action-hz 50 --max-latency-ms 200 \
  --clients 1 --robot-prefix waiter --warmup 1 --duration 60 \
  --output data/multi_robot_experiment/waiter.json
```

应该先看到 `waiter-0: waitlisted`。占位者启动约 31 秒后实验自动结束，等待者收到第二条注册响应 `accepted`，随后开始接收动作，期间不需要重启服务端。

这轮等待者前半段在等资源，其 `delivered_hz` 用整个测量窗口作分母，因此会低于 5 Hz。查看最后注册状态和动作到达间隔，不能把等待时间计入后得到的平均值误判成调度降频。

正常退出测试的是 WebSocket 断开释放。心跳超时可以用暂停客户端进程（Linux `SIGSTOP`）模拟：先在单独进程运行一个获准客户端，记录其 PID；对该 PID 执行 `kill -STOP <PID>`，等待超过 3 秒，观察等待者晋升；最后 `kill -CONT <PID>` 并结束该进程。只操作自己启动的客户端，不要暂停 Server。网络失联会有额外检测延迟，不要求恰好 3.000 秒。

## 10. 汇总结果与判定

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path('data/multi_robot_experiment').glob('sweep_*.json')):
    result = json.loads(path.read_text())
    robots = list(result['robots'].values())
    active = [robot for robot in robots if robot['status'] == 'accepted']
    waiting = [robot for robot in robots if robot['status'] == 'waitlisted']
    events = sum(sum(robot['events'].values()) for robot in robots)
    print(path.name, 'connected=', result['clients'], 'accepted=', len(active),
          'waitlisted=', len(waiting), 'total_delivered_hz=',
          round(sum(robot['delivered_hz'] for robot in robots), 2),
          'active_hz=', [round(robot['delivered_hz'], 2) for robot in active],
          'events=', events)
PY
```

至少保留以下实验记录：

| 项目 | 内容 |
| --- | --- |
| 软件 | Git 提交、Python/JAX/PyTorch 版本、推理后端 |
| 硬件 | GPU 型号、驱动、显存、是否有其他 GPU 任务 |
| 模型 | checkpoint、输入规格、horizon、去噪步数、精度 |
| 服务端 | 实测 mean/P95/P99、budget、准入容量、调度参数 |
| 客户端 | 连接数量、accepted/waitlisted、目标 Hz、延迟约束 |
| 结果 | 每台 delivered Hz、动作到达间隔、丢弃原因、暂停情况 |

建议先使用 60 秒窗口找边界，然后选 N 个客户端做更长稳定性实验，例如 `--duration 600`。公平性应逐客户端比较，不能只看总吞吐。

一次完整跑通应满足：CPU 演示和回归测试通过；GPU 单客户端返回正确形状的动作；多客户端稳定共享单模型；超容量连接进入等待；断开后能晋升；JSON/日志可追溯。是否满足真实机器人的控制需求，需要额外的网络和闭环实验，本手册不作该结论。

## 11. 常见故障

| 现象 | 检查方式 |
| --- | --- |
| 找不到 load_test.py | 当前只有旧提交，尚未同步本手册的配套脚本 |
| Connection refused | 模型仍在加载/profiling，或端口/地址不对；看终端 A 日志 |
| 远程机器连不上 | 客户端 URI 使用服务端可达地址，检查监听端口和防火墙；`127.0.0.1` 只适用于同机 |
| Policy/input spec mismatch | 客户端 policy-id 必须为 `pi05_droid`，input-spec 与 Server 一致 |
| Observation shape/dtype differs | 两端使用同一份 sample；不要把 float64 状态在客户端改成 float32 |
| Action horizon cannot cover | 当前 horizon 与 action_hz 无法覆盖整个推理周期，不是 GPU 连通性问题 |
| Latency budget cannot cover | `max_latency_ms` 小于 lead time + service budget，先做合理预算再注册 |
| 大量 missing_observation… | 观测拷贝/编码/传输超出 lead time，或 Worker 忙；检查网络和单客户端表现 |
| deadline_exceeded | 实际推理比准入预算慢，检查 GPU 竞争和模型输入；不要仅增大连接数 |
| suspended | 连续预算超限或推理错误；查 Server 日志，解决后重新启动与 profiling |
| 内存/线程开销随客户端增加 | 本负载器用同进程异步连接和线程回调；大量客户端可能先受负载机限制，可分多机并使用不同 robot-prefix |

客户端脚本没有随机采集抖动、自动多机部署、端到端延迟埋点或实际动作执行器。先完成这里的可重复基线，再逐项加入这些因素，避免把不同实验混为一个性能数字。

## 本手册的验证范围

CPU 演示、现有测试以及新增负载脚本的回环重放可在无 GPU 环境验证。GPU 权重下载、目标机器上的真实模型运行和多机网络实验需要在对应设备执行，不能将 CPU 模拟结果当作这些检查已经通过。
