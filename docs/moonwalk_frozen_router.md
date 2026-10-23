# 太空步冻结通用策略验证版

**结构范围更正（2026-10-23）：本页及 G2 是两个完整 Actor 的外部隔离对照，
不能作为用户要求的“同一 HoloMotion 内部扩展专家”的完成证据。
当前主方案、待验收项和后续状态见 `docs/moonwalk_internal_moe.md` 与
`test_outputs/internal_moe_expansion_progress.json`。以下保留历史结果，不改写实验数据。**

本轮已完成 G2 的限定范围仿真验证。最终结果与边界见
`docs/moonwalk_router_G2_validation_20261023.md`，独立模型包在
`test_outputs/sliding_router_G2_review_20261023/`。定时跟进已停用。
复现最终版本应使用包内 `config.yaml` 和 `model_G2.pt`；下方默认训练配置用于早期
专家学习，不是 G2 的最终评测配置。

本实现位于 K1 项目 `holomtion_boosterK1`。初始实现未启动训练；后续自主推进状态见
`test_outputs/sliding_router_progress.json` 和具体任务的 `job.json`。
入口配置是 `training/motion_tracking/motion_tracking_booster_k1_moonwalk_frozen_router`。

## 结构与冻结边界

```text
506 维原 Actor 输入 → 冻结的原观测归一化
                       ├→ 原完整 MoE 策略（冻结）→ μ_general
                       ├→ 独立完整滑步策略      → μ_specialist
                       └→ 白名单参考特征 → 上层 gate

最终输出 → 原有 22 维动作和 PD 接口
```

滑步专家是原策略的完整副本，包含独立 Attention、8 个路由专家、共享专家和输出头。
它不会进入原模型的 Top-2 专家竞争。以增加一份策略计算和缓存为代价，先建立明确的隔离对照。
两个分支在每步都处理真实历史输入，即使没有被选中，避免切换时使用过期 KV cache。
环境 reset 和 rollout 边界同时清除两个分支的对应缓存，序列重算使用相同的 episode mask。

冻结项包括通用分支所有参数、其内部 router、观测归一化统计和原动作噪声参数。
通用分支始终 `eval()`，使用 `no_grad()`；正常训练模式切换不会解除这些冻结。
原动作噪声由两个分支共用并保持不变，验证版不支持 sigma override。
Critic 按原 PPO 流程独立训练，不参与部署动作输出。

上层 gate 使用原 `router_obs_terms` 白名单中的参考特征，包括未来参考窗口。
它不额外读取仿真接触真值、摩擦真值、动作名称或实际脚滑速度。
部分参考坐标量沿用原模型的机身坐标表达，不能称为绝对与机器人状态无关。
新 gate 可通过 `sliding_router.zero_feature_columns` 屏蔽这些列；屏蔽只影响上层
分类器，不改变任何旧策略输入或内部路由。早期 D 单独屏蔽未达到识别要求；
G2 配合校准、静止负例和时序确认在本轮参考上通过评测，仍无未见滑步泛化保证。

## 路由模式

| `modules.actor.sliding_router.mode` | 行为 | 用途 |
| --- | --- | --- |
| `base` | 直接返回原通用输出 | 旧能力保持对照，仅评测 |
| `specialist` | 直接返回滑步分支输出 | 默认阶段 A，只训练专家 |
| `soft` | `μ_general + sigmoid(gate) × (μ_specialist − μ_general)` | 可微 gate 的 PPO 训练和同模式评测 |
| `hard` | 按阈值选择完整分支，默认阈值 0.5 | 自动选择评测；无 PPO gate 梯度 |
| `temporal_hard` | 连续高置信度后进入专家，连续低置信度后退出 | 带迟滞的自动选择，仅评测 |

软模式是单个 Gaussian 的均值插值，沿用冻结 sigma，并非两个 Gaussian 的混合分布。
rollout 和 PPO 重算使用相同公式，不使用直通梯度或训练/评测不同的路由规则。
软混合在 gate 非零时会改变旧动作；冻结原分支本身不保证学习型路由不误选。
`base` 模式，以及 `hard` 选择通用分支时，直接返回原输出，不经过插值运算。
`hard` 仍为逐帧选择；`temporal_hard` 通过 `enter_steps`、`exit_steps`、
`threshold`、`exit_threshold` 实现逐环境连续确认与迟滞，不插值控制输出。
它的运行状态随两分支缓存同时 reset，加载权重也会清空；状态不保存到 checkpoint。
序列评测从新前缀开始，按 attention mask 的 episode 边界重置，且不修改在线状态。
它未接入携带初始门控状态的 PPO rollout 重算，因此明确禁止用于 PPO 训练。
迟滞减少抖动，也引入进入和退出延迟；连续物理切换仍须单独仿真验证。

输出 TensorDict 额外包含 `slide_gate_probability` 和实际使用的 `slide_gate_weight`。
二者在强制路由下可以不同；概率是分类器结果，weight 才是实际控制选择。

## 权重初始化、保存与恢复

默认从完整通用 `stage_3_complex/model_11999.pt` 初始化，未使用退化的 `model_3998`。
只有 `bootstrap_from_general=true` 配合 `checkpoint_load_mode=finetune` 时，才允许
把旧 Actor 的完整 state dict 严格复制到两个分支，并初始化新 gate；优化器从新状态开始。
缺失旧网络参数仍然报错，不使用 `strict=False` 掩盖不兼容。

原生保存流程仍生成 `model_N.pt` 和对应的 `model_N/actor`、`model_N/critic`。
新的 Actor 权重同时包含两个分支、gate、冻结归一化及噪声参数。
恢复验证版时指定新的 checkpoint，设置 `bootstrap_from_general=false`，并按需要设置
`checkpoint_load_mode=resume`；完整复合 checkpoint 的任意缺失 gate/分支权重都会报错。
路由模式、阈值与训练开关沿用项目惯例保存在运行配置中，恢复时必须使用相同配置。
切换训练阶段会改变优化器参数集合，应使用 `finetune` 创建新优化器。

## 不启动训练的检查

在本项目根目录执行：

```bash
source train.env
PYTHONPATH="$PWD" "$Train_CONDA_PREFIX/bin/python" \
  tools/booster_k1_retarget/preflight_sliding_router.py \
  --report test_outputs/sliding_router_preflight_20261023.json
```

此检查只在 CPU 构造网络并读取真实权重，核对数据根目录，比较合成输入上的输出；
不启动 Isaac、不采集 rollout、不更新参数。报告不能解释为太空步已学会或机器人效果通过。

只打印未来训练命令：

```bash
HOLOMOTION_TRAIN_DRY_RUN=1 \
CONFIG_NAME=motion_tracking_booster_k1_moonwalk_frozen_router NUM_ENVS=64 \
  bash holomotion/scripts/training/train_motion_tracking.sh
```

默认配置继续使用已有太空步片段、脚底姿态/速度与接触奖励，保持 `feet_slide=0`。
训练和验证指向同一合成片段，属于训练诊断；正式评估仍需独立太空步参考和旧动作回归集。
默认训练配置未改变摩擦设置；实际启动记录以对应的 `job.json` 为准。

## 自主推进与共享显存

2026-10-23 用户授权在审查通过且资源允许后自主启动训练。定时跟进每 15 分钟推进一次，
读取 `test_outputs/sliding_router_progress.json`，避免重复训练。完成时停用跟进。

`tools/booster_k1_retarget/sliding_router_job.py` 默认只检查资源；带 `--run` 才启动
8 环境、最多 20 次迭代的小规模试跑。它要求至少 10 GiB 空闲并连续复查，保持 4 GiB
共享余量，本任务预算为 6 GiB；每 2 秒监测，低于余量或超过预算时只停止自己创建的进程组。
使用文件锁限制本任务同时一个作业，不停止或调整其他人的训练。该预算是监测阈值，
不是 NVIDIA/PhysX 的硬隔离，其他进程突然增大占用仍可能抢占余量。

启动还要求有效的 `test_outputs/sliding_router_audit.json`；审查相关代码一旦改变，
哈希检查会拒绝沿用过期审查。后续扩大规模前必须重新测量峰值，不能从网络参数量推算仿真显存。
这里的短试跑只验证运行和资源占用，不代表太空步任务通过。
测量并核对短试跑后，可用 `--phase specialist` 进行最多 500 次迭代、16 环境、
每环境 32 步的有界学习；必须提供已完成且冻结检查通过的 `--previous-pilot` 和对应
`--checkpoint`。`--phase evaluation` 使用同一显存监控，复制配置并链接只读权重，
在独立输出目录执行原生评测，不覆盖训练目录。当前已完成的运行记录见进度文件。

```bash
source train.env
PYTHONPATH="$PWD" "$Train_CONDA_PREFIX/bin/python" \
  tools/booster_k1_retarget/sliding_router_job.py \
  --output-dir test_outputs/sliding_router_pilot_resource_check
```

检查不会启动训练，也不会更改其他人的进程。

本轮审查修复：重新加载 checkpoint 会同时清空两分支旧 KV 缓存；Adam 和 AdamW
优化器均只接收允许训练的参数。新增测试覆盖实际逐环境硬切换、缓存回绕、分块推理、
局部及全局 reset，以及非零 gate 下的参考特征隔离。

原生评测包装器会把 gate 使用的归一化参考列保存为 `gate_samples.npz`；标签只用于
训练监督，不作为 gate 输入。`fit_sliding_gate.py` 只在 CPU 更新上层 gate，保存完整的
复合权重，并再次验证所有控制器张量未变。schema-v2 记录器依据当前每环境 `_clip_indices`
获取实际动作名，并保存环境、批次、参考帧号、分类概率和实际控制选择。旧记录器按批次
原始行顺序命名，短动作结束重新分配后可能标错来源分组，原 C/D 分类保留组结果降级为
开发诊断；未来 gate 拟合拒绝这些旧格式，必须重新采集。切换合成片段的监督标签设为 -1，
防止把整段误当普通动作或滑步，使用它训练前必须提供分段任务标签。
普通动作按来源动作分组保留；正例来自同一
合成片段，使用带边界间隔的时间分块，仍不属于未见滑步动作测试。当前阶段的已查看
保留组只能用于开发诊断，最终结论需要新的独立动作。
CPU gate 快照保留旧控制器的训练元数据，只用于评测；继续优化时应显式设置训练开关并
使用 finetune 创建新优化器，不能恢复原专家优化器作为 gate 优化器。

## 后续 gate 阶段

阶段 A 先验证独立专家能学会太空步，gate 固定旁路。之后冻结专家，用包含普通动作负例
和太空步正例的观测数据训练 gate，标签表示期望动作意图，不由实际打滑生成。
Actor 提供 `gate_supervision_loss(obs_td, slide_labels)`，使用冻结归一化和相同白名单特征，
不执行控制器、不改变缓存；损失是 BCE with logits。调用前设置 `train_gate=true`、
`train_specialist=false`，由调用者只优化 gate 参数。该辅助接口尚未自动接入在线 PPO 标签流。

也可使用 `soft` 配合 `train_gate=true` 进行 PPO，冻结或共同更新专家由
`train_specialist` 控制。必须另行准备混合动作采样及正确的按动作目标奖励；
当前阶段 A 的单太空步数据配置不应直接当作通用 gate 训练配置。

至少分别评估：强制通用的旧动作回归、强制专家的太空步表现、学习型路由的正负例识别、
普通动作与太空步之间的连续切换。报告旧动作完成率、gate 误选率、支撑稳定性、
滑动脚速度与左右交替，不能只报告训练 reward。

## 验证边界

此版本支持原生 PyTorch Actor/PPO 接口和双分支缓存，禁用自动编译及训练后的自动仿真钩子。
ONNX/真机导出暂不支持，调用导出会明确报错，避免只导出一个分支造成行为不一致。
CPU 单元测试覆盖严格加载、优化器冻结边界、gate 梯度、参考特征白名单、缓存 reset、
分块推理与 PPO 序列重算。已完成专家训练及 G2 自动滑步、旧动作保持、连续切换仿真。
具体覆盖、原模型已有失败和未验证边界以最终报告为准；未验证多 GPU、导出或真机。
