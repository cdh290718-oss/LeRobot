# LeRobot 实验进度与 Agent 交接（2026-09-20）

> **接手先读本文。当前基线为 10k + BF16；离线推理、双相机只读推理及两次 10 秒受限实机执行已完成。肘关节跟随不足与启动扭矩异常尚未解决，旧 `Execute` 暂停使用。下一步是接收/分析只读电机诊断回传，不是放宽保护或直接做完整抓取。**
>
> 本轮用户要求整理并上传交接资料，没有要求自动连接 Windows、操作机械臂或启动新训练。沿用“提供脚本 → 用户本机运行 → 回传 ZIP → 开发机分析”的方式。仓库维持私有；新的 agent 需要用户授权的仓库访问权限。

## 1. 两分钟掌握当前状态

| 阶段 | 状态与结果 | 证据 |
|---|---|---|
| 数据采集/审核 | 0918 的 67 段、35,651 帧；56 段训练/11 段验证；0916 的 25 段因相机布局不同未混入 | [原始项目交接](交接文档.md)、[数据来源](../training/episode_provenance.json) |
| 开发机微调 | 从 200 步继续到总计 10k；6k 验证 loss 0.0763，10k 0.0915；已完成 | [训练日志](../training/remote-records/logs/train10k.summary.log) |
| Windows 离线评估 | 6k/10k × FP16/BF16 四组合全部加载与有限值检查通过；33 个独立样本、每组合 100 次真实 chunk 测量 | [离线复核](../experiments/lerobot_local_deployment_20260919/windows_review_113316/REVIEW_zh.md) |
| Windows 双相机只读 | 两路约 30 FPS；60 秒 Live 有 76 组 50×6 预测，电机零写入；原状态轮询实际约 22.46 Hz | [实时只读复核](../experiments/lerobot_live_readonly_20260919/review_121541/REVIEW_zh.md) |
| Windows 受限实机 | 两次完整 10 秒，各 295 控制周期，约 29.57 Hz；结束保持回读通过 | [return1 复核](../experiments/lerobot_bounded_execution_20260919/review_return1/REVIEW_zh.md) |
| 当前问题 | 两次约 1.7–1.8 秒因肘关节跟随不足停止；另有首次启动扭矩变化无法解释、旧代码清理覆盖不足 | [问题与轨迹](../experiments/lerobot_bounded_execution_20260919/review_return1/ANALYSIS.json) |
| 只读电机诊断 | 已交付并做无硬件测试；截至本次盘点，尚未找到用户诊断回传 | [诊断说明](../experiments/lerobot_bounded_execution_20260919/review_return1/motor_diagnostic_kit/README_zh.md) |
| 完整自主抓取评估 | **尚未完成**；本轮夹爪固定、动作大量限幅，不能报告抓取成功率 | 不要把训练 loss、离线误差或 10 秒控制通过当作抓取成功 |

0918 文档中“尚未开始本地推理”的描述是历史状态。0919 `PLAN_zh.md` 也仅是当时方案；最新状态以本文和实际回传为准。EgoSMPLX / Sapiens2 / WiLoR 是另一任务，不纳入本次 LeRobot 交接。

## 2. 已验证环境与输入语义

### Windows 本机（由用户操作）

- 根目录 `G:\LeRobot`；使用 `G:\LeRobot\.venv\Scripts\python.exe`，Python **3.12.14**，torch **2.10.0+cu128**，LeRobot **0.6.0**，Transformers **5.5.4**。
- RTX 5060 Laptop 8 GB；驱动用户报告 582.05。系统 PATH 中 Python 3.14 / torch 2.11 是另一环境，本任务不要切过去。
- 当前 editable 源码：`G:\LeRobot\source\lerobot-0.6.0\src\lerobot`，有原项目修改。不要用上游最新版覆盖。
- OpenCV 为 headless 包，旧实时预览用本机 HTTP 页面；不要假设 `cv2.imshow` 可用。
- 原环境未安装 pip，不代表模型不能运行；无需因此重建现有环境。
- 机械臂 follower **COM5**，leader **COM6**；设备配置 `G:\LeRobot\data\robots\lerobot.json`。
- 校准：`G:\LeRobot\data\calibration\robots\so_follower\lerobot.json`。只适用于这台设备，当前脚本要求与归档精确一致。
- HF 缓存：`G:\LeRobot\cache\huggingface`。
- 已部署微调模型：`G:\LeRobot\offline_kit\models\006000\pretrained_model\` 和 `...\010000\pretrained_model\`。

| 视角 | 原逻辑名 | 归档设备索引 | 模型键 |
|---|---:|---:|---|
| 外部 | 3 | 1 | `observation.images.camera1` |
| 腕部 | 1 | 2 | `observation.images.camera2` |

两路采集均 640×480、30 FPS、MJPG；实际 Windows 后端记录为 DSHOW。USB 重插后索引可能变化，应看实际画面，不能把逻辑名当索引。保存 config 中列出的 camera3 不表示要伪造第三路图像。

固定任务文本：`Pick up the yellow cube and place it into the mesh container.`

状态和动作顺序均为：`shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos`。前五维为本项目校准角度，夹爪为 0–100 归一化值；不是末端 XYZ 位姿。保留 checkpoint 的预处理、归一化/反归一化，不自行套用其他单位。

原 LeLab rollout 入口有未显式传递 `rename_map` 的源码问题；自定义工具明确传递相机映射。`use_amp=true` 也不等于已选 BF16，自定义工具使用明确的 `torch.autocast(..., dtype=torch.bfloat16)`。

### 开发机

SSH 别名历史为 `myserver`，是否可连接取决于接手者自己的授权环境。训练根目录：

```text
/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918
```

环境为 `venv/bin/python`，Python 3.12.13、torch 2.7.1+cu118；GPU 为共享 A100-SXM4-80GB。训练与测试应使用独立环境、限制资源，不结束其他任务或修改 GPU 全局设置。后续只做资料分析时无需启动 GPU。

## 3. 模型、训练与评估结果的正确解释

本轮仅使用 0918 数据；验证段为 24 号最后 1 段和 25 号全部 10 段，即合并后的 episode 56–66。不是随机按帧划分，也未验证跨场景泛化。33 个离线样本是每个验证 episode 抽 3 帧，不能称作 33 段。

训练采用 `lerobot/smolvla_base`，约 450M 总参数、约 100M 可训练参数；冻结视觉编码器，训练动作专家/状态投影。正式训练从 200 步恢复到总计 10k，北京时间 0918 17:57:34–19:08:56，约 71 分 22 秒。最终训练 loss 0.127；采样验证 loss 0.0915；6k 的 0.0763 是这次记录最低值。训练采用 batch4、worker2、BF16、线程2、nice19、每步让出0.30秒等共享资源限制。详见现有 training 记录，不重复训练。

Windows 离线指标：

| 配置 | 完整模型 chunk P50/P95（ms） | 生成动作归一化 MSE |
|---|---:|---:|
| 6k FP16 | 843.9 / 897.3 | 0.309093 |
| 6k BF16 | 784.7 / 828.5 | 0.309772 |
| 10k FP16 | 414.3 / 928.7 | 0.245006 |
| 10k BF16 | 748.9 / 817.7 | 0.245605 |

10k 在这批生成动作误差上比 6k 低约 20.7%，与训练的 flow-matching 验证 loss 是不同指标；不能推导抓取成功率提升。当前选择 **10k BF16** 为暂定部署基线，保留 FP16 对照，不声称 BF16 总是更快。

实时只读 Live 完整 pipeline P50/P95 为 **898/992 ms**。受限执行回传的两次成功试验变为 **383/388 ms 左右的 P50**；改善原因未做受控定位，不能把差异归因为权重结构变化或已证实的电源问题。PyTorch 峰值 allocated 约 **928 MiB**、reserved **958 MiB**，不等于整卡或整个 Windows 进程总显存。

每次模型调用产生 **50×6** 动作，训练时间基准 **30 Hz**、覆盖约 **1.67 秒**。控制循环约 29.57 Hz 不代表每秒做 29.57 次模型推理；不要把读取动作缓存的耗时算成网络速度。

## 4. return1 的关键发现与待解决问题

共 10 包：1 次 DryRun、7 次 Execute、2 次 Release。CRC、Python 代码 SHA256 均通过；所有存在的预测数组形状和有限值检查通过。

- **13:01:10、13:02:28**：各运行 10 秒、295 周期，结束保持目标与六路扭矩开启均回读成功。
- **13:04:03、13:06:40**：分别记录 49/53 周期后触发 4°跟随误差。肘关节目标相对起点变化 −4.308°/−4.659°，实测仅 −0.264°/−0.615°。最后有效周期误差 3.956°，随后保持读数相对最后目标差 4.044°。触发异常那一轮未保存，不应冒充有精确失败周期记录。
- **12:55:22**：第一条 Goal_Position 对齐写入后，Present_Position 读取报 `Incorrect status packet`；代码尚未执行显式 Torque_Enable 写入，退出却读到六路扭矩全为 1。现有证据不足以判断固件行为、其他控制来源或通信读回异常。
- 旧 `may_be_enabled` 在显式启用前才设 True，漏掉首次目标写入后的异常清理。已准备候选补丁，把“可能已通电”的标记前移，并对齐后核对扭矩；**仅两个针对性回归测试通过，未实机验证、未并入旧执行包**。
- 用户对接触/断电/释放/重摆历史回答“应该没有，我忘了”，现场操作应记为未确认。有些相邻包之间的扭矩变化不能从日志还原。
- 原标定 `wrist_roll` 为 **2042–2051，仅 9 tick（约0.79°）**。本轮保留并限幅，不能自行扩大或重标定。

旧执行器仅用于本次 10 秒小幅试验：相对启动位置最多 ±8°并与已有范围取交集，每次目标最多 1 tick，夹爪固定；模型独立进程、父进程独占串口，按观察时间跳过已过时动作。这是简单时间对齐，不是完整 RTC。控制失误差、帧龄、循环卡顿、动作计划过期均会停止推进。退出通常保持扭矩，`Release` 是另一条显式命令。USB/进程故障下的软件保持不构成硬件急停保证。

实际成功运行也存在大量限幅（肘关节约96.6%/99.7%周期），不能代表未经限制的策略行为或完整抓取成功。

## 5. 接手后的优先任务

1. 检查用户是否新回传 `*_MotorDiagnostic_return.zip`；本次盘点 `cui_local_computer_files/results/` 时尚无此包。若已回传，先校验 CRC、源代码哈希及只读包统计。
2. 读取六路已有模式、PID、限矩、速度、目标/实测位置、负载、电流、电压、温度及状态字，重点比较 `elbow_flex`。当前诊断保存 SDK 原始值，不能在没有具体固件/规格确认时随意换算电流电压。
3. 分开排查肘关节跟随不足、串口异常、启动扭矩变化。静止诊断不能排除运动时问题；不要把猜测写成根因，也不要简单放宽 4°阈值、速度或角度限制。
4. 下一版需同时审查：启动候选修正；异常周期数据落盘；停止回读和状态不确定处理。候选修正不是可直接批准完整抓取的版本。
5. 若后续用户要求再次实机评估，先据诊断确定具体改动，再单独交付有新版本号的工具。保持已有模型、相机语义和校准；别用旧 Execute 反复复现异常。
6. 完整抓取、夹爪动作、固定次数成功率、6k/10k 公平实机对比和多任务泛化都仍待开展，不能写成已完成。

当前可交给用户的下一步是 **只读诊断**，不是执行修正候选：

- [只读诊断 ZIP](../experiments/lerobot_bounded_execution_20260919/review_return1/motor_diagnostic_kit.zip)
- [本机运行说明](../experiments/lerobot_bounded_execution_20260919/review_return1/motor_diagnostic_kit/README_zh.md)
- [启动修正 diff（供审查）](../experiments/lerobot_bounded_execution_20260919/review_return1/startup_fix_candidate/startup_cleanup.diff)

在本机关闭其他控制程序、保持稳定支撑后，解压到 `G:\LeRobot`，命令为：

```powershell
cd G:\LeRobot\motor_diagnostic_kit
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-MotorDiagnostic.ps1
```

约 15 秒，只读，不启用/解除扭矩、不发送动作。回传生成的 ZIP，包括失败结果。**本段是交接说明，不表示接手 agent 应自动运行硬件。**

## 6. 开发机原始文件地址与 GitHub 副本

下表为绝对路径；在 GitHub 上这些开发机地址不会自动变成可下载资源。已上传副本使用右栏链接。其他全部归档文件的开发机绝对地址、仓库相对地址、大小和 SHA256 见 [逐文件清单](experiment_archive_manifest_20260920.json)。

| 资料 | 开发机绝对地址 | GitHub 副本 |
|---|---|---|
| 本文开发机原稿 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_handoff_20260920/HANDOFF_zh.md` | 本页 |
| 初始部署方案（历史） | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_local_deployment_20260919/PLAN_zh.md` | [方案](../experiments/lerobot_local_deployment_20260919/PLAN_zh.md) |
| 离线代码/33样本 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_local_deployment_20260919/offline_kit/` | [目录](../experiments/lerobot_local_deployment_20260919/offline_kit/) |
| Windows离线复核 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_local_deployment_20260919/windows_review_113316/REVIEW_zh.md` | [报告](../experiments/lerobot_local_deployment_20260919/windows_review_113316/REVIEW_zh.md) |
| 只读实时工具 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_live_readonly_20260919/live_readonly_kit/` | [目录](../experiments/lerobot_live_readonly_20260919/live_readonly_kit/) |
| 只读实时复核 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_live_readonly_20260919/review_121541/REVIEW_zh.md` | [报告](../experiments/lerobot_live_readonly_20260919/review_121541/REVIEW_zh.md) |
| 旧受限执行器（暂停） | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_bounded_execution_20260919/execution_kit/` | [归档代码](../experiments/lerobot_bounded_execution_20260919/execution_kit/) |
| return1详细复核 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_bounded_execution_20260919/review_return1/REVIEW_zh.md` | [报告与图](../experiments/lerobot_bounded_execution_20260919/review_return1/REVIEW_zh.md) |
| 只读电机诊断 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_bounded_execution_20260919/review_return1/motor_diagnostic_kit/` | [目录](../experiments/lerobot_bounded_execution_20260919/review_return1/motor_diagnostic_kit/) |
| 启动修正候选 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_bounded_execution_20260919/review_return1/startup_fix_candidate/` | [候选代码](../experiments/lerobot_bounded_execution_20260919/review_return1/startup_fix_candidate/) |
| 3份离线/实时原始回传 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/cui_local_computer_files/results/` | [原始ZIP目录](../cui_local_computer_files/results/) |
| 10份return1原始回传 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/cui_local_computer_files/results/return1/` | [原始ZIP目录](../cui_local_computer_files/results/return1/) |

本机回传历史包名：`20260919_113316_962_Run_return.zip`、`20260919_121454_372_Inputs_bf16_return.zip`、`20260919_121541_727_Live_bf16_return.zip`，以及 return1 下按时间命名的 10 包。保持原始 ZIP 不变，有利于核对每次用户实际运行版本。

### 未上传 GitHub 的大资源

| 资源 | 开发机绝对地址 |
|---|---|
| 10k完整模型 | `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/outputs/train10k/checkpoints/010000/pretrained_model/` |
| 6k完整模型 | `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/outputs/train10k/checkpoints/006000/pretrained_model/` |
| 训练状态/其他checkpoint | `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/outputs/train10k/checkpoints/` |
| 合并训练数据 | `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/data/ChomCUI/yellow_cube_0918_v1/` |
| 数据来源映射 | `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/data/episode_provenance.json` |
| 训练完整日志 | `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/logs/train10k.log` |
| 开发机环境 | `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/venv/bin/python` |
| HF基础模型缓存 | `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/cache/huggingface/` |
| 含权重离线交付包 | `/vepfs-mlp2/mlp-public/huyaoqing/EgoFishPose/experiments/lerobot_local_deployment_20260919/lerobot_offline_full.zip` |

每个完整 pretrained_model 共7个文件：主 `model.safetensors`、`config.json`、两个 processor JSON、两个归一化 safetensors、`train_config.json`。主权重每份 906,712,520 bytes。GitHub 的 JSON 元数据不是完整模型；不要只下载 JSON 后声称可离线推理。

基础 VLM：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，已测试固定 revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`。当前构造器仍需本地基础权重/processor 缓存。微调文件校验值见 [checkpoint_manifest.json](../experiments/lerobot_local_deployment_20260919/offline_kit/checkpoint_manifest.json)。不上传模型、优化器、完整视频、虚拟环境或认证凭据。

## 7. 归档结构、可复现性与测试边界

本次把开发机实验目录结构原样映射到仓库 `experiments/`，原始回传映射到 `cui_local_computer_files/results/`，因此相邻文件引用和原始回传分析尽量保持可用。33 个 NPZ 仅为离线验证样本，不是完整训练数据。

- 原受限执行代码曾通过18项控制/故障测试，以及 A100 上真实10k BF16模型的模拟控制联调；这些测试没覆盖本次暴露的首次写入异常情形，不能当作硬件全通过。
- 只读诊断主程序正常/读取失败两条路径已测试；尚待用户实际电机诊断回传。
- 启动候选两个针对性回归测试通过；不是整包重验，也不是实机修复验收。
- 旧交付 README 中可能保留当时的“下一步 Execute”指引；**当前操作边界以本文为准**。历史代码和压缩包保留原始哈希，不暗中改写。
- 部分开发机 build/export 脚本包含原机绝对路径，并非 GitHub checkout 后任意环境可直接重建。只读分析需要 NumPy；图表重建还需 Pillow/Matplotlib。
- [return1分析脚本](../experiments/lerobot_bounded_execution_20260919/review_return1/analyze.py) 可使用已归档原始 ZIP 和代码副本重新核对数据。不要把重建报告变成执行模型/硬件的入口。

## 8. 可直接发给下一位 Agent 的任务摘要

> 请先读 `docs/LeRobot实验进度与Agent交接_20260920.md`，以它为最新状态入口，再读 return1 的 REVIEW_zh.md 和 ANALYSIS.json。SO101/双RGB/SmolVLA已经完成0918数据10k微调、Windows离线与只读实时推理，以及两次10秒受限实机执行。当前采用10k BF16，Windows现有环境为G:\LeRobot\.venv的Python3.12.14和torch2.10cu128。肘关节跟随不足、启动时未解释的扭矩变化和首次写入异常清理仍需处理；不要继续旧Execute、放宽保护或重校准。用户通过本机运行脚本并回传ZIP协作，不连接其电脑。优先查找新的MotorDiagnostic回传，先做只读复核。权重在开发机 `/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/outputs/train10k/checkpoints/`，其余开发机绝对地址和已归档文件校验在本文及manifest中。完整抓取成功率还未验证，任何继续实机动作或训练应依据用户后续要求。
