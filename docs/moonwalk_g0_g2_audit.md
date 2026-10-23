# 内部 MoE：G0–G2 补缺与证据

后续代码修订见 [内部准入与同布局重算审计](moonwalk_admission_replay_audit.md)：新增硬 gate 和受约束 cached PPO，310 项 CPU 回归通过。下文保留上一轮 e 版证据；任意批形状与自动 gate 行为仍不能宣称通过。

2026-10-23。范围：任务书 v2 对照、代码修复、CPU 单元/接线测试、实际 checkpoint 迁移、已有观测记录重放。没有启动训练、新仿真、自动推进或真机，也没有提交/推送。

**结论：固定 sliding 任务的奖励、Critic、GAE 和 rollout 合同已接通；旧参数和受保护 legacy 路径有保持证据。自动 expanded 路由下的旧动作保持仍未通过，G0–G2 不能整体标为完成。** 任意批量重算还存在近 Top-2 并列的数值边界，已加入更新前的强制检查。新版本没有训练，不能宣称学会太空步或连续切换稳定。

## 原架构与真实基线

- 工作仓库：`/home/tcl/Desktop/start/holomtion_boosterK1`。基线为 `logs/HoloMotionK1SequentialCurriculum/overnight_20261020_153041_Bl3s1P/stage_3_complex/model_11999.pt` 及其 sidecars。
- 一个 Actor、一层 MoE、8 个旧细专家、1 个共享专家、2 个新增细专家，实际配置的 Top-2 和 softmax 打分保留。一套 Attention、输出头和 KV cache；没有额外完整 Actor 或新动作头。任务书中的 sigmoid 只是示例，不能替代实际模型配置。
- 新增 Actor 可训练参数 197122，独立新专家 bank 和新增 router 参数；优化器按 Parameter identity 白名单构造。旧 bank、共享专家、Attention、reference encoder、输出头、Actor normalizer、sigma、旧 router 和前向相关状态冻结。
- 两个独立 Critic，分别对应旧奖励与滑步奖励；每个专家不另建 Critic。旧 Critic 全冻结，新 Critic 可训练。Actor、旧 Critic、新 Critic 的参数和 buffer storage 互不共享。
- 旧奖励完整配方与 model_11999 相同。滑步配方在独立副本增加 sole_pose=2、sole_velocity=1.5、heel_contact=1。**原 11999 的 feet_slide 已为 0**，本次没有通过额外关闭此项改变旧任务。
- 新专家复制旧专家 0/1 的 FFN 权重但不共享存储。新增路由初值/预热仍按登记配置；无预热的零更新路由不选新专家只说明保守初值，不能当作新能力。

## 本轮补齐的执行缺口

1. **完整任务合同。** 不可变 TaskSpec 绑定 task/reward/critic、完整奖励与实现、Actor/Critic schema 和原模型配置、command/reference、终止、动作/PD/仿真配置、XML/URDF 内容、gamma/lambda 和 timeout 语义。额外核对原 Actor 配置，阻止 head 数、normalizer epsilon 等“权重形状不变但计算改变”的修改。
2. **真实参考内容。** 同时固定 HDF5 v2 manifest 和其中实际 clip 引用的 shard；支持旧子集 manifest 指向外部 shard，覆盖 clip 切片与身份变更。验证阶段限制每任务最多 128 个被引用文件、256 MiB；超限须另做登记，不能静默略过。当前 sliding 2 个文件、legacy 11 个文件（其中各含 manifest）。
3. **奖励记录。** 每项保存 raw、weighted_rate、weighted_step；实际加权和 dt 仍由原 manager 应用一次。零权重项由原 manager 跳过，记录 validity，不伪造原始值。PPO 检查启用项齐全且分量之和等于收到的 reward。raw sampled action 与处理后的 joint target 分开记录，后者在自动 reset 清空 action buffer 前捕获；log_prob 始终对应原高斯采样变量。
4. **Critic 与统计。** 只读 registry 可按存储 task_id 诊断取值。live Stage A 仅允许同质 sliding，旧或混合任务 PPO 明确拒绝。新 Critic 输入统计在 rollout 与全部 PPO epochs 期间冻结，周期结束后按该任务的全 rank sum/sumsq/count 更新一次；旧统计保持不变。当前没有额外 return/value-output normalizer。
5. **GAE。** 先沿真实 `[T,E]` 时间轴计算，再组 minibatch。bootstrap mask 与 trace mask 分开；episode/task 边界检查独立存在。滑步合同将 timeout 定义为有限时域终止，b=0、m=0；不使用 reset 后状态，也不拿当前 value 冒充 final value。continuing truncation 纯函数要求显式 pre-reset final_values，已有反例测试；真实环境的 continuing-task final observation 路径仍未接入。
6. **rollout 来源。** 保存 task/reward/critic、policy/route profile/revision、normalizer、TaskSpec digest、episode/timestep、有效标记、terminated/truncated、reward components、next bootstrap value。当前 storage 要求全为真实、未 padding 的 transition；GAE 后的分任务归一化 helper 支持 valid mask 和固定顺序的 DDP collectives。
7. **采样和重算一致性。** Actor 参数及归一化 buffer、两个 Critic 全部参数/buffer 的 identity/version 在采样周期内受检查。更新前验证完整 rollout 和即将使用的同一随机 permutation 下全部 minibatch；真正 update 复用这些索引。任一 rank 检出 mu/sigma/log_prob 超差，所有 rank 在 optimizer step 前报错。容差预先固定为 mu `atol=5e-5, rtol=1e-5`，sigma `1e-6,1e-5`，log_prob `1e-4,2e-5`，没有为失败样本放宽。
8. **KV 与恢复。** route/stage/warmup 改变即提升 route revision 并清 KV；更新后也清 KV，不能把不同版本的历史静默混用。完整恢复核对任务、参考内容、路由、实现代码与优化器状态。保存显式注册的 legacy/sliding TaskSpec，允许用已登记旧任务合同离线加载新 bundle；训练 resume 仍严格要求原 active task 与 registry。旧单 Critic 的 model_199 不能直接作为新版恢复点。

新增 balance/inactive 利用率压力限定在滑步 Stage A 的新增 bank；旧专家、旧路由不接受这些梯度。Stage B 的冻结接口和历史 CPU replay 测试不等于已完成自动协调训练。

## 当前证据与任务书对应

最终产物以以下版本为准：

- `test_outputs/internal_moe_g0_g2_20261023_e/`：audit、param_manifest、migration、baseline_metrics、resolved_config、task_specs、零更新 checkpoint、实现快照和验证来源清单。
- `test_outputs/task_isolation_g0_g2_preflight_20261023_c/report.json`：真实 model_11999、原 PPO 加载入口、10 次合成 CPU Critic 优化器更新。
- `test_outputs/internal_moe_g0_g2_regression_20261023_e.log`：**195 passed**。包括 CPU Gloo 双进程统计测试和小模型优化器测试，均非环境训练作业。

| 验收项 | 结果与范围 |
|---|---|
| U01–U02 白名单及 storage | 通过；新 Actor 参数与新 Critic 独立，旧对象不进优化器 |
| U03–U04 梯度因果 | 通过；value loss 不更新 Actor，detached advantage 改值可改变新增 Actor 梯度 |
| U05–U07 奖励与 GAE | 通过固定任务接线、跨任务扰动及手算边界；混合任务 live PPO 未开放 |
| U08–U09 初始化与 legacy 等价 | 通过严格迁移和登记范围内的缓存/短历史对照；48 个旧 Actor 状态保持 |
| U10–U11 反例 | 通过：高分新专家确实能扰动旧行为；grad mask 不能阻止 packed AdamW decay |
| U12–U13 更新与梯度 | 合成 CPU 1/10 次更新旧参数/buffer 不变；实际模型新增专家和 router 有有限有效梯度 |
| U14 序列/缓存 | **条件通过，非全通过**：正常 cooperative/reset/回绕/分块测试通过；真实旧记录存在一帧批形状敏感近并列，见下文 |
| U15–U17 因果、恢复、缓存版本 | CPU 测试通过；实际 zero bundle 完整 PPOTF resume（含优化器 LR）和已登记 legacy 离线 load 通过；修改 gamma 拒绝 |
| U18–U19 DDP与错配 | CPU 两 rank 缺任务样本仍同序归约；reward/critic/task/version/normalizer 错配拒绝；没有运行 GPU DDP |
| U20 环境合同 | 与保存的旧配置逐项比较通过，当前资产与参考内容已固定；无法由当前文件哈希反推 2026-10-20 的全部资产内容 |

G0 已提供配置、参数、数据和来源审计；行为基线复用历史单 seed 的 11 段旧动作（10 个原始 motion keys），不扩展为未见动作保证。G1 固定 sliding 路径具备可执行隔离合同。G2 参数/迁移/legacy 保护有证据，但 U14 的一般性数值一致性与自动 expanded 保持未通过，不能据此放行训练。

## 旧动作是否受影响：三种证据不能混写

| 对照 | 结果 |
|---|---|
| 旧参数冻结 | Actor 48 个旧状态张量、Critic 19 个旧状态张量逐字节不变 |
| 受保护 legacy | 真实 5555 帧缓存 CPU 重放对历史 GPU 输出，mu 最大差 `2.86102294921875e-6`，Top-2 选择零帧不一致；历史闭环对照 11/11 完成、匹配关节轨迹逐字节相同 |
| 零更新 expanded、关闭预热 | 相同旧历史中新专家选择 0 帧，对 legacy mu 最大差 `1.9073486328125e-6`；没有学习发生 |
| 历史单 Critic Stage A model_199 expanded | 旧状态仍逐字节相同，但 5067/5555 帧选中新专家；平均绝对 mu 差约 `0.005947`，最大 `0.1250601`，平均 KL 约 `0.004437`（22 维动作求和） |

最后一行是**相同输入历史下的路由漂移证据**，来自旧实验 checkpoint；不是新版奖励/Critic 隔离训练结果，也不是新做的闭环摔倒率结论。旧路由保持仍需要之后的普通动作负例与双侧锚点协调，并做真实旧任务/过渡闭环验证。

## 两项没有掩盖的序列边界

1. 旧记录第 4171 帧的两个候选 router 分数仅差约 `1.86e-9`。CPU 批量 GEMM 的末位舍入改变 Top-2，造成 `0.0155697` 的 mu 差。逐条缓存重放对齐，但任意批形状对齐不能宣称成立。audit 同时保留原批量失败和单历史诊断结果；PPO 预检会拒绝超差，不复用采样的旧 Top-K 来掩盖问题。
2. 原 legacy `sequence_mu` 在超过 context 的 rolling mask 下把首个可见 token 当成 episode 起点。小 CPU 反例误差可达约 `0.22876`。验证版明确拒绝这一组合，保留 native 完整序列和短历史路径；需要缓存推理或 context 内打包历史。未改旧 router 数学或原网络文件来制造“全等”。

profile 改变后清缓存是版本保护，不是连续动作过渡方案。站立→滑步→站立、保持物理状态的任务切换、多种子和新版技能效果均未验证。

## 复查方式与产物状态

从仓库根目录 `source train.env`，用该环境 Python 运行 `tools/booster_k1_retarget/audit_internal_moe_g0_g2.py --output-dir <新目录>` 或 `preflight_task_isolation.py --output-dir <新目录>`；工具只做 CPU 读取、合成测试与序列重放，不构造仿真器。输出目录必须新建，避免覆盖旧证据。

本轮审计 a 为早期诊断，b/c 在合同/manifest 接线处失败，d 为上一版通过快照；均保留，**不能拿它们替代 e 的最终实现哈希**。更早的 156 项报告是先前版本记录。本轮没有采用附件附带的玩具测试结果作为本仓库证据。

自动任务 `automation` 保持 `PAUSED`；本任务没有 GPU 作业。后续工作的先决条件仍是用户新的训练授权，以及上述未通过门槛得到处理；当前代码不会自动恢复推进。
