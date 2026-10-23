# 太空步任务隔离：实施审计与范围

2026-10-23。用户授权检查、整理后修改；禁止启动训练或恢复自动推进。

后续 G0–G2 补缺、195 项回归、真实 checkpoint 完整恢复与旧动作重放见 [当前审计](moonwalk_g0_g2_audit.md)。下文保留上一版 156 项证据及当时边界；完整 TaskSpec、分任务 DDP 统计、rollout 版本和缓存合同已在后续补上，混合任务 live PPO 与自动旧行为保持仍未完成。

## 改动前缺口

现有内部扩展保留一个 Actor、8 个旧细专家和 2 个新增细专家。旧参数、Actor 归一化和 sigma 已冻结，但奖励仍是一份平面配置，Critic 是单个可训练 PPOCritic。

与实际 model_11999 的完整配置比较，仅增加 sole_pose=2、sole_velocity=1.5、heel_contact=1 三项；原 checkpoint 的 feet_slide 已经为 0，不能声称本次通过关闭该惩罚才改变旧目标。现有 TaskGatedReward 使用高层 holo_task_ids，仓库没有为它建立滑步/普通动作标签，缺失标签还会静默返回零，不能直接用于本任务。

## 本轮实施顺序与合同

1. 建立 CPU 可测试的奖励 profile 选择器：从完整配置独立复制 legacy/sliding 配方，旧配方剔除且仅剔除三项滑步奖励，校验其哈希等于实际 model_11999。参数、dt 和权重仍由原 IsaacLab reward manager 处理一次，不再套额外乘法。
2. 增加 FrozenLegacySlidingCritic：旧 Critic 全状态冻结，新增 Critic 独立存储、独立归一化；仅新增 Critic 进入优化器。初始从经哈希登记的原 Critic 严格载入，禁止把已微调过的 Critic 当旧模型。完整恢复要求版本、合同和全部键一致。
3. 第一版每次运行固定一种任务，Stage A 只允许 sliding on-policy PPO。奖励 profile、Critic 和真实 motion raw key 必须一致；任务编号写入 rollout 的观察 TensorDict，训练重算从存储读取，不修改 Actor 输入布局。普通任务允许 legacy 只读评估。混合任务 PPO、episode 内换任务明确拒绝，不用全局优势归一化伪装分任务实现。
4. 新任务显式定义有限时域 timeout 为终止（不 bootstrap）；旧任务不改终止/物理/执行合同。当前上游用当前 value 近似 timeout bootstrap，尚无 final observation，因此不沿用该近似或冒充正确 truncation。未来支持 continuing-task truncation 必须增加 final observation 路径和测试。
5. 冻结白名单、奖励扰动隔离、Critic 梯度/统计隔离、任务错配、严格恢复及真实 checkpoint CPU 检查通过后，仅交付代码与证据。不会启动 1/10/100 update 或仿真。

## 附件的使用边界

`HoloMotion_Incremental_MoE_Taskbook_v2_Package.zip` 作为审查参考，已阅读任务书、manifest、玩具测试源码和附带结果。其测试不是本仓库的验收证据；其中试训、8 个新增专家、多个种子、G4 混合任务和 git commit 要求不自动成为本轮指令。本轮沿用已登记的 2 个新增专家；用户“保留旧 Critic”解释为全冻结旧 Critic，优先于附件允许继续训练旧 Critic 的可选项。

## 验证边界

本轮 CPU 合同测试无法证明滑步学成、自动路由保持旧行为或站立→滑步→站立稳定。历史 11 个旧动作 legacy 完整保持是原隔离路径证据，不能替代 expanded 自动协调验收。新奖励的物理定义沿用现有三个函数，不悄悄更换滑步指标。

## 已实施文件

- `holomotion/src/motion_tracking/incremental_task.py`：独立复制完整奖励配方、固定任务标签检查、真实 raw motion population 检查、状态/配置/实现哈希。任务编号只用于训练与价值选择，没有加入 Actor 或新增路由的输入。
- `holomotion/src/env/isaaclab_components/isaaclab_rewards.py::build_rewards_config`：仅在显式配置 incremental_task 时选择配方并启用错配保护。保留原 callable 签名与顶层 SceneEntityCfg 参数解析；权重与 control_dt 仍由原 manager 应用一次。普通配置走原路径。
- `holomotion/src/modules/incremental_critic.py::FrozenLegacySlidingCritic`：一个全冻结 legacy PPOCritic 和一个独立 sliding PPOCritic。权重、normalizer 无共享存储；强制载入已登记原 Critic 后才能 forward。新 Critic 暂沿用原 843 维输入，独立训练，不新增 Actor 动作头。
- `holomotion/src/algo/ppo_tf.py`：可选 Critic 工厂、过滤冻结参数的优化器、储存 task_id、GAE 前检查和 replay 检查、仅新任务的有限时域 timeout 处理、严格初始化与恢复。保存/核对完整任务、路由 schedule 和关键代码哈希；不匹配时拒绝恢复，不静默跳过优化器状态。
- `holomotion/config/training/motion_tracking/motion_tracking_booster_k1_moonwalk_task_isolated.yaml`：新增专用配置，原配置和已有实验保留。默认从 model_11999 新初始化，不能将此前单 Critic 的 model_199/model_100 直接冒充新版恢复点。
- `tests/test_incremental_task_isolation.py`、`tools/booster_k1_retarget/preflight_task_isolation.py`：真实仓库接线测试和实际 checkpoint 的 CPU 预检。

旧任务只读评估需同时选择 legacy 任务、正确旧数据集与 legacy 路由；配置任务和实际动作来源不符会报错。Stage A 训练入口拒绝 legacy/mixed 任务。正式 resume 必须将 Actor/Critic 的 bootstrap_from_legacy 都设为 false，保留 checkpoint 内登记的任务与预热设置。关闭预热后重新训练属于显式阶段迁移，本轮没有实现迁移工具；离线评估允许关闭预热、改变路由消融模式。

## 本轮 CPU 证据

完整预检：`test_outputs/task_isolation_cpu_20261023_d/report.json` 与同目录 `resolved_config.yaml`。实际经过 PPOTF 配置检查与原 checkpoint sidecar 加载路径，但没有构造 SimulationApp。

| 项目 | 结果 |
|---|---|
| 回归 | 156 passed，原始输出在 `test_outputs/task_isolation_regression_20261023.txt`；其中新增隔离文件 16 项 |
| 真实旧奖励 | 与 model_11999 全配方相等，改变滑步项系数不改变旧配方与固定输入旧 reward tensor |
| 旧 Critic | 10 次合成 CPU AdamW 更新后 19 个状态张量逐字节不变；仅新 Critic 参数/统计变化 |
| 旧 value | 同一未分块路径最大绝对误差 0；分块 GEMM 最大绝对误差 2.384185791015625e-7 |
| 旧 Actor | 48 个状态张量保持；70 步缓存、reset/回绕对照误差 0；新增专家/路由有有效梯度 |
| 梯度因果 | 独立 value loss 不产生 Actor 梯度；改变 detached advantage 符号会改变新增 Actor 参数梯度方向 |
| GAE | 原 storage 手算终止/reset 边界一致；混合 task 存储在 bootstrap 前拒绝；新任务 timeout 不使用当前 value 冒充 final value |
| 恢复 | 错误来源、缺键、错误版本/合同、预热变化、关键代码变化、优化器组/动量形状不匹配均测试拒绝 |
| 环境合同 | 原关节顺序、动作数、actuators、默认角度、termination、domain_rand、simulation 配置及 Actor/Critic schema 逐项比较相同；新增 timeout 回报语义仅作用 sliding |
| 静态检查 | 四个新增 Python 文件 Ruff check/format 通过；git diff --check 通过 |

最初预检 a 的数值断言通过但配置导出失败，已保留 `export_failure.json`，不能把它当完整预检；b/c 为迭代中的检查，最终以 d 为准。附件玩具测试未执行，没有将附带的通过记录计入 156 项。

## 尚未完成与禁止推论

任务书 G0–G2 没有全部完成。本轮实现的是固定任务 Stage A 的奖励/Critic 接线和 CPU 合同保护，不是完整的混合任务 CriticRegistry/分任务 DDP 统计系统。

尚未运行 IsaacLab、on-policy PPO 控制 smoke、独立旧动作闭环、无预热新技能验收、Stage B 双侧锚点协调、贡献消融、多种子统计或站立→滑步→站立闭环。持续任务 truncation 的 final observation bootstrap、完整 TaskSpec 所有环境/参考版本字段、跨 policy 版本的通用缓存合同仍需要后续工作；本轮遇到不支持的任务或恢复情况会拒绝，不降级训练。

附带任务书允许未来 1/10/100 update 并不构成用户授权。当前没有启动训练、没有恢复自动推进、没有提交或推送；自动任务仍为 PAUSED。
