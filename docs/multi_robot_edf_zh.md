# 多机器人 EDF 与观测预取

在本分支用 `--multi-robot` 启动 OpenPI 时，默认启用柔性预约、非抢占 EDF 和 JIT 观测预取。单个 Policy 实例仍通过一个 Worker 执行 Batch=1 推理。旧固定时隙实验可用 `--scheduler-mode FIXED` 复现。

## CPU 对照实验

从已安装依赖的 OpenPI 根目录执行。示例的模型调用模拟约 30 ms，客户端观测准备模拟 12 ms，二者均可在 CPU 上跑通。

```bash
.venv/bin/python examples/multi_robot/demo.py --mode EDF --no-prefetch \
  --clients 3 --duration 10 --capture-delay-ms 12
.venv/bin/python examples/multi_robot/demo.py --mode EDF \
  --clients 3 --duration 10 --capture-delay-ms 12
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q src/openpi/serving packages/openpi-client
```

`Granted Hz` 按注册响应列出各客户端获准频率，0 表示等待。`Prefetch.requests` 统计推理期间发起的采集，`ready_while_busy` 统计上一次推理结束前已收到的观测，`used` 统计最终用于推理的预取观测。关闭预取时三项均为 0。比较相同负载下的动作块数量和上述计数，延长窗口可以减少首尾效应。

## 部署和注册

复用现有 [观测样本和 checkpoint 准备流程](multi_robot_experiment_zh.md)，替换服务端启动命令：

```bash
.venv/bin/python scripts/serve_policy.py --multi-robot --scheduler-mode EDF \
  --profile-observation data/multi_robot_experiment/droid_sample.msgpack \
  --input-spec '{"num_cameras":2,"history_frames":1,"image_size":[224,224]}' \
  --scheduler.max-utilization 0.85 --scheduler.lead-time 0.015 \
  --edf.max-collect-time 0.03 --edf.max-observation-age 0.10 \
  policy:checkpoint --policy.config pi05_droid --policy.dir /path/to/checkpoint
```

阶段一单独运行时加 `--edf.no-prefetch`。收发协议沿用 `register`、`collect`、`observation`、`actions`、`heartbeat`、`dropped`。EDF 的 `register_ack` 回传 `scheduler: edf`，`slot_id` 为 `null`，`assigned_hz` 和 `period_ms` 表示实际预约。等待者获准后会收到新的 `register_ack`。两个客户端可以声明不同频率，但仍要匹配同一个 `policy_id` 和输入规格。

`load_test.py` 可以通过 `--hz`、`--min-hz` 指定目标和最低频率。用不同的 `--robot-prefix` 同时启动多个进程，即可在同一个 Server 上比较异频客户端，或将启动时间错开验证运行中加入。旧 Ferrata 调度客户端要求 `assigned_hz` 等于目标，使用它时把 `min_hz` 设为 `target_hz`。

## 预约、截止时间和预取

- 每台机器人预约一个周期 `1 / assigned_hz`。每周期释放至多一个新请求，非抢占 Worker 按截止时间选择可执行的请求。客户端上下线时只增减预约，已有客户端保持原预约频率和下一次释放时间。
- 准入先尝试目标频率，再尝试目标与最低频率之间可行的值。动作覆盖长度 `action_horizon / action_hz` 也会提高有效最低频率。容量不足时等待，最低要求连单独运行都无法满足时拒绝。调度器同时检查总利用率、周期内需求和非抢占阻塞，准入时将整段采集预算计入成本，尽管启用预取时实际可以重叠。
- 请求截止时间是 `释放时间 + min(预约周期, max_latency)`。当前阶段使用这个预约截止时间，客户端实际 Action Buffer 深度未纳入计算。发给客户端的频率承诺基于预算内的观测准备和推理，实际过期请求会被报告并丢弃。配置 `--edf.max-collect-time` 时应使用实测采集、编码和网络往返数据。
- 预取开启时，Worker 运行期间从已经释放的请求中预测下一台，按剩余推理时间和该客户端近期 `collect → observation` 的 P95 估计发采集通知的时刻。Worker 完成或观测到达后重新决策。晚到、重复、过期的观测会被淘汰，错过的周期不会补跑。观测年龄限制从服务端发出的采集通知计时，客户端仍需提供实际新鲜的帧。
- 持续超出预约容量时沿用 `suspended` 通知。CPU 实验能验证时序、会话隔离和 Worker 串行执行。真实 GPU 的容量、网络交付延迟和动作缓冲表现需要在目标部署上测量。

附件方案里的动作缓冲区反馈、服务债务加权和多步序列 Lookahead 属于后续阶段。这里的预测只用于把下一次观测准备与当前推理重叠。
