# 太空步隔离版复审修复与源码发布（2026-10-23）

本轮只修改代码并执行 CPU 测试。未启动训练、模拟器、GPU 作业或恢复自动推进。用户随后明确授权创建 GitHub 私有仓库 router-k1 并推送源码。

## 六项复审结论

1. **DDP buffer 同步误报已修复。** 参数仍严格检查 identity/version；buffer 在版本变化时比较逐字节快照。相同数值的广播通过，实际数值变化、替换、布局变化拒绝。CPU 单进程 Gloo DDP 真 forward 已覆盖。缓存重算版本仍限制单 rank FP32，不能将此解读成完整 DDP PPO 已验证。绕过版本计数的原始存储写入、未被观察到的修改后原样恢复不在检测保证内。
2. **观测合同已补齐。** TaskSpec 绑定完整 env.obs 和 normalization，包括函数、参数、历史、噪声及启用开关；同时登记场景传感器、环境间距、机器人资产配置及 XML/URDF/STL 内容。仅排除设备放置、输出目录和执行资源元数据。当前资产支持范围限定为本任务本地 K1 XML/URDF/STL，不支持的变体或依赖明确拒绝。
3. **旧动作自动路由入口已接通。** legacy/sliding 各自登记 legacy_control 与 automatic：分别是强制旧路由/closed gate 和 cooperative/learned gate，后者关闭 warmup，仅允许离线评估。它们使用同一任务奖励、Critic、参考人口和观测配置，不从路由预测决定任务标签。固定环境/算法 seed=42；同步顶层、环境机器人及 command 的 motion 配置副本。实际评估加载器保留登记的物理/噪声配置，compile_config 使用 eval=True，避免覆盖源训练配置。设备编译不再造成合同误报。
4. **启动器默认配置已修复。** internal 默认选 moonwalk_admission，继承 task_isolated 双 Critic/奖励隔离；旧单 Critic 必须显式选 historical_single_critic。CPU --review-config 检查真实 Hydra 结果。审核记录必须匹配实际启动合同、代码、配置及 checkpoint sidecar 哈希；旧审核记录不能直接放行新版。
5. **恢复轮次已修复。** checkpoint 明确保存最后完成轮次与下一轮次；正常完成 n 后从 n+1 恢复，零更新从 0 开始；旧格式有明确兼容解释。拒绝覆盖已有 checkpoint/sidecar。完成标记放在子类更新收尾之后。
6. **恢复学习率已修复。** 恢复优化器及用于 adaptive 调度的 actor/critic 基础学习率，校验各组 lr_multiplier。缺失或不一致的恢复状态在加载权重前拒绝。finetune 保持新优化器；offline 禁止进入 learn。

## CPU 验证

- 相关完整回归：**406 passed in 26.15s**，日志为本地 `test_outputs/incremental_review_repairs_full.log`。
- 覆盖奖励/任务 GAE 隔离、冻结状态、原生/新增专家路由、缓存重算/回绕/reset、checkpoint、实际 Hydra 启动器/评估入口、观测/资产合同反例。
- 新改动的独立模块和测试 Ruff 通过；PPO/PPOTF 语法编译通过。
- 本轮早期检查过真实 11999 初始化与 5555 帧旧记录（`test_outputs/internal_moe_review_repairs_20261023_a`）；随后补强了配置/资产合同，因此该旧零更新产物不作为最终源码恢复合同证据。历史冻结与强制关闭 gate 的动作一致性证据见 `moonwalk_admission_replay_audit.md`，不能外推为新版自动 gate 已验收。

## 尚未验证

没有训练自动 gate，也没有执行新版滑步学习、混合任务仿真或连续站立→滑步→站立。当前能证明合同/隔离及 CPU 反例已覆盖，不能宣称自动路由下旧动作完全不受影响。新的严格观测/实现合同与旧产物不同；不允许静默恢复旧扩展 checkpoint，需要显式重新迁移并审核。

提交到新仓库的是源码、配置、测试、必要 K1 静态资产与说明；本地 checkpoint、参考数据、运行日志与环境凭据不随源码发布。部分实际配置/集成回归需要外部 11999、参考子集及本地训练依赖，空仓下载后需补齐这些资源。原工作目录和原 Git 远程保持原样。
