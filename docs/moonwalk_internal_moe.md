# 太空步：同一 Actor 内部增加细专家

2026-10-23 用户纠正结构层级。前一版 G2 是两个完整 Actor 的外部选择对照，
不能作为本任务“内部扩展专家”的完成证据；历史代码、模型和评测数据全部保留。
当前任务进度以 `test_outputs/internal_moe_expansion_progress.json` 为准。

## 当前实现

入口：`training/motion_tracking/motion_tracking_booster_k1_moonwalk_internal_moe`。
实现：`holomotion/src/modules/internal_moe.py`，PPO factory 已接入。

本地 model_11999 是单层 MoE：8 个细专家、1 个共享专家、Top-2，506 维观测、22 维动作。
扩展后仍只有一个 Actor、一个 Attention 层、一套 KV 缓存和一个原动作输出头。
新增 2 个细专家（中间特征 FFN），新增打分线性层复用原冻结的参考特征嵌入；
没有扩展 router 的观测白名单，没有外部动作控制器或动作均值插值。
原白名单中的部分参考坐标依赖实际机器人朝向，仍不能称为完全与机器人状态无关。

原参数保持原 state_dict 键名、独立 Parameter 存储，全部冻结。
新增 FFN gate/up、down 和路由打分 weight/bias 共 197122 参数；
旧 Actor 含噪声参数共 2305762 参数。运行时拼接专家权重用于原 grouped-MM，
优化器只能接触新增参数，旧参数没有梯度也不受 AdamW 衰减影响。
输入编码、参考嵌入、Attention、共享专家、8 个旧细专家、旧 router、输出头、
辅助头、原归一化统计和 sigma 均不更新；旧 expert_bias、计数、EMA 使用状态不更新。

`legacy` 直接执行原 MoE 计算，新增专家在 softmax/Top-2 之前完全不参与。
`cooperative` 拼接旧、新打分后原生 Top-2，支持旧＋旧、旧＋新、新＋新。
`new_only` 仅供消融，屏蔽旧细专家；共享专家和冻结主干仍在，不能称为移除所有旧知识。
自动合作模式的旧动作保持必须实测，冻结本身不提供这一保证。

## 训练阶段和约束

Stage A 仅用于新动作数据，训练新增 FFN 和新增路由参数。
试跑预热按同一参考输入确定性地提升一个新增专家的分数，保证进 Top-2；
rollout 与 PPO 重算同一公式，没有随 forward 改变的计数或随机门控。
闲置专家 margin loss 另教未提升的原始新分数进入 Top-2，防止只在预热中有响应。
balance loss 只对新增专家之间的条件分布/利用率计算，闲置 margin 只拉新增专家；
不要求旧、新 10 个专家均匀工作。这些压力仅在 Stage A cooperative 新任务上启用。
预热必须明确关闭后再评测正式的 cooperative 能力，预热表现不能当作自动路由成功。
试跑 8 环境 × 8 步 × 20 迭代，仅用于验证管线/梯度/冻结/资源。

Stage B 的冻结开关已实现：所有专家冻结，只剩新增路由参数可训练，关闭利用率压力。
`fit_internal_moe_router.py` 已实现 CPU 有序窗口的双 teacher 动作分布 KL 和路由 KL
校准，明确需要普通动作负例和新动作记录，过渡数据须指定新动作参考帧区间。
单帧不能脱离真实历史，缺前缀、乱序或不连续的 episode 位置会报错。
工具已通过小模型保存/严格恢复及冻结边界测试；实际双侧保持和仿真验收尚未完成。
PPO 入口明确拒绝把 Stage B 当普通新任务 PPO 运行。
两个 teacher 仅用于训练/评估，最终部署仍只有扩展后的一个 Actor。

## 缓存和序列重算

第一版显式限制为实际 checkpoint 的单层 MoE。Attention 位于专家之前，
切换 `legacy/cooperative/new_only` 不分叉、不清空 KV；模式变化只清除路由历史诊断。
episode reset 和加载 checkpoint 清除原生缓存；分块计算复用同一批环境的缓存视图。

新测试发现原生 sequence_mu 在传入滑动窗口 mask 时，用每行第一个可见位置推导
RoPE episode 起点，窗口回绕后与在线绝对位置不一致。扩展版 cooperative 重算
改为从相邻 token 的 episode 连通性确定 reset，独立应用固定上下文窗口。
legacy sequence 原路径保持不变，用于严格同执行方式对照；legacy 缓存推理也完全保留。
不能拿原生长序列重算的历史问题当作本扩展正确性的证明。

原始 checkpoint 仅允许显式 bootstrap + finetune，添加新增键，其余旧键/形状仍严格校验；
已有扩展 checkpoint 缺新键、版本不符均报错，不以 strict=False 容错。
恢复已有扩展模型须关闭 bootstrap，保存配置中的 stage/mode/warmup 也是复现条件。
暂未验证 ONNX/显式外部缓存导出，入口拒绝导出；没有操作真实机器人。

## 已有证据与待验收项目

实际 checkpoint CPU 检查：原 48 个状态张量在新增参数 AdamW 更新后逐位不变；
legacy 缓存执行 70 步，包括 32 长度回绕和部分环境 reset，均值最大误差 0，sigma 完全一致。
CPU 回归覆盖新旧三种 Top-2、任务梯度、Stage B 参数边界、严格初始化/恢复、
全批/分块、episode reset、回绕、采样 log-prob 与序列重算一致、模式切换共用 KV。
截至首次试跑前，相关回归 131 项通过。检查文件/结果见独立 JSON，不代表仿真成功。

首次 20 轮试跑峰值 3102 MiB，保存后 48 个旧状态张量全部不变，但关闭预热后
新增专家 0 次选中，不接受为能力成功。随后仅对新增路由 514 个参数设置 300 倍
学习率（适配学习率更新仍保留倍率，新增 FFN 维持 3e-6），完成 200 轮 × 16 环境
× 32 步的 Stage A 继续训练，峰值 3132 MiB，旧状态仍全部不变，训练时两个新增专家
均有参与。尚须以无预热仿真判断技能，而不是以利用率代替效果。
含 Stage B 有序回放保存/恢复与学习率隔离的相关回归现为 136 项通过。

待验收：Stage A 无预热学得可测太空步机制；对比初始、legacy、cooperative、new_only；
旧动作 rollout 保持；Stage B 有序旧/新/过渡回放和双侧保持；站立→滑步→站立闭环，
含接触代理、追踪、完成率、动作突变和推理开销。没有通过前不得宣布任务完成。

## 资源和输出

复用 `sliding_router_job.py --architecture internal`，必须提供本任务独立审查 JSON；
与历史外部方案共用同一作业锁，避免双任务同时占卡。每次 GPU 启动前持续 10 秒
观察至少 10 GiB 空闲，自身最多 6 GiB，保留至少 4 GiB 空闲；每 2 秒监测，
不足只停止守护程序创建的本任务进程组，不影响其他人的作业、不预占显存。
单环境评测通过 `evaluate_internal_moe.py` 保存完整有序原始 Actor 观测、
episode 位置、参考帧、专家选择/权重与动作分布，供后续同历史 teacher 校准。
现有输出不覆盖，不提交或推送代码。
