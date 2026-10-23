# 内部专家准入与 PPO 同布局重算验证

2026-10-23。本轮只修改代码、运行 CPU 回归、加载真实 checkpoint 和重放已有观测。没有启动训练、新仿真、自动推进、真机，也没有提交或推送。

**结果：内部硬准入和受约束的 PPO 重算路径已补齐。gate 强制关闭时，5555 帧旧观测的动作均值与原模型同路径 CPU 推理逐位相同。自动 gate 的识别准确率、新版滑步能力、站立→滑步→站立的闭环稳定性仍未验证。**

## 实现与任务隔离

- 同一个 Actor：1 层 MoE、8 个冻结旧细专家、1 个冻结共享专家、2 个新增细专家，保留实际模型的 softmax 和 Top-2。Attention、reference encoder、输出头、原 router、sigma、normalizer 和原辅助预测头保持冻结。
- 新增线性 gate 仅读取原有 reference hidden，没有 task_id、奖励或训练标签输入。`closed/open/learned` 返回硬布尔准入，没有动作软混合或直通梯度。
- 关闭时直接调用原候选集合、原归一化和原专家 kernel。混合 batch 中对完整原 batch 计算 native 输出，再按硬 mask 选择，避免子集 GEMM 形状变化。新专家在归一化与 Top-2 前被排除。
- Stage A 仍只允许同质 sliding 任务，gate 强制 `open` 且 gate 参数冻结，只训练新增 FFN 和新增评分 router；独立滑步 Critic 按滑步奖励训练。Actor 可训练参数仍为 197122。
- Stage B 接口冻结专家，训练新增 router 与 gate。新增可选、按正负类别均衡的 BCE，强制同时提供普通动作负例与滑步正例，保留新旧行为/路由锚点。标签只进入监督损失。本轮没有执行 gate 拟合。
- balance/inactive loss 只统计 Stage A 中准入的有效 token，只对新增 bank 施加压力。普通动作闭门 token 和 padding 不被要求激活新专家。重算汇总全部时刻及 chunk，避免只用最后一个 chunk 的辅助损失。
- 沿用上一轮的独立完整奖励配方、冻结旧 Critic、独立滑步 Critic、任务合同及 GAE bootstrap/trace 边界。11999 的 `feet_slide` 原本就是 0；新增三个奖励仍只属于滑步配方。混合任务 live PPO 仍拒绝。

## 重算、缓存与原辅助头

新配置为 `holomotion/config/training/motion_tracking/motion_tracking_booster_k1_moonwalk_admission.yaml`，继承 task-isolated 配置。

`sequence_replay=cached` 在临时 KV 上按采样时相同的单步路径重算。强制环境数量、环境顺序和 chunk 大小一致；固定 `num_mini_batches=1`，不打乱环境，多个 PPO epoch 仍保留。每次真实重算新参数下的 Top-2，不复用采样时的选择。

支持完整 episode 因果 mask 或其精确 context 窗口、异步 reset、padding 和缓存回绕；非法稀疏/未来/跨 reset mask 拒绝。成功或异常都恢复原 live KV 对象及内容、router shift 状态、统计开关和临时 hook。显式改变 route/stage/warmup/gate mode 会提升版本并清 cache。同层 gate 按输入自然开关不改变 Attention 前端，不需要两套分支缓存。

原方法名 `return_pre_moe_hidden` 在此单层结构中实际返回该层 FFN 后的 hidden。本次捕获 `norm_f` 输入，保留原辅助头语义和它到新增专家的梯度，没有删除辅助头来绕过 checkpoint 匹配。

更新前 mu/sigma/log_prob 检查继续存在，容差没有扩大。CPU 同布局 mu/sigma 可逐位一致；原采样与序列 log_prob 公式约有 1e-6 舍入差，仍使用原 `atol=1e-4, rtol=2e-5` 合同。

## 首版明确限制

1. **单 rank、禁用 AMP。** CPU bf16 反例发现原生分块函数会把 chunk KV view 转为独立张量，丢掉原 cache 写入。采样和重算可能同时错误但相互一致，因此本版 setup/runtime/Actor 入口显式拒绝 autocast。
2. **不支持完整 DDP PPO。** 两进程 CPU 反例确认默认 DDP buffer 广播即使内容不变也会修改 `_version`，触发状态 lease。cached 路径显式拒绝多 rank。既有 GAE/统计纯函数的 CPU Gloo 测试不能证明完整 DDP PPO 可用。
3. 只支持已登记的一层结构。ONNX、编译及 GPU 最终数值/吞吐量未验证；新配置继承禁用编译和导出。
4. 任意 batch 重排/切分仍不保证一致。上一轮第 4171 帧的近 Top-2 并列反例仍有效；本次限制执行布局，没有修改旧 router 数学。

## 初始化和恢复

带 gate 的扩展版本为 v2。历史无 gate 的 v1 可读，但不能隐式恢复成 v2。显式 native bootstrap 时，新 FFN/router 权重来自旧权重，新 router bias 与 gate 参数/元数据从配置重新初始化。回归覆盖污染这些值后再次 bootstrap，防止残留上一次新能力状态。

gate 的维度、阈值、初始化偏置和版本有独立 digest。完整 PPO 恢复继续检查 TaskSpec、实现哈希、路由合同和优化器。离线评估仅可切换 gate mode，不能顺带改阈值；不能用 `strict=False` 静默补齐 gate。

Stage B 工具当前输出仍是明确标记的评估/finetune 产物：复制源 `.pt` 元数据并更新 Actor sidecar，不能直接作为 PPO resume checkpoint。本轮没有执行或生成这类训练产物。

## 证据

| 检查 | 结果 |
|---|---|
| 相关 CPU 回归 | **310 passed**；`test_outputs/internal_moe_admission_regression_20261023_b.log` |
| 真实 native bootstrap/零更新保存 | 11999 Actor/Critic 通过；完整 PPOTF optimizer 恢复与登记 legacy 离线加载通过 |
| 旧状态 | Actor 48 个状态保留；原 Critic 19 个状态通过冻结/独立存储合同及回归 |
| 真实旧记录 | 5555 帧、11 段 episode、E=1 原环境布局 |
| closed gate 对同路径 native CPU | mu 最大差 **0**，Top-2 差异 **0**，选择新专家帧数 **0** |
| 对历史 GPU 记录 | mu 最大差 **2.86102294921875e-6**，Top-2 差异 **0**，sigma 最大差约 **5.96e-8** |
| 对抗准入 | 移植历史新增 bank 并将新评分 bias 设为 1e6，closed gate 仍保持上述结果 |
| 真实模型辅助头反向 | 7 项原辅助输出保留；新增 FFN/router 有非零有限梯度，冻结参数无梯度，所有状态不变，optimizer step=0 |
| 小模型边界 | 混合开关、softmax/sigmoid/new_only、长序列/reset/回绕/chunk、全序列辅助 loss、异常清理通过 |

最终审计：`test_outputs/internal_moe_admission_20261023_d/`，含 `audit.json`、配置、参数/迁移清单、任务合同、实现快照、`verification_manifest.json` 和未训练的 `model_zero.pt`。零更新 checkpoint 从 11999 初始化，不含对抗测试在内存中移植的历史 bank。

审计 a 发现原辅助头启用后提前拒绝；b 完成主体检查；c 的附加探针因工具属性名写错退出。这些目录保留，不替代 d。历史 bank 移植有来源记录，不作为新版训练或恢复证据。

复现：加载 `train.env`，用项目 Python 运行 `tools/booster_k1_retarget/audit_internal_moe_admission.py --output-dir <新目录>`。工具不构造仿真器、不执行优化器更新、不启动 GPU 作业。

**验收：G0/G1 固定任务隔离证据保留；G2 在单 rank、全精度、同布局和强制闭门保护范围内补齐。自动 gate 的旧动作误开率/滑步召回率、新版技能机制和连续切换仍未完成，整项任务不能标为成功。** 自动推进继续暂停，训练仍待用户另行授权。
