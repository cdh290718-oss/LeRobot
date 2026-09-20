**SO101 / SmolVLA Windows 本地推理部署方案（2026-09-19）**

结论：优先复用 G:\LeRobot 现有 Windows 原生环境和源码，部署本项目已经微调的 6k、10k 完整 checkpoint。RTX 5060 Laptop 8GB 具备进行 batch=1 本地推理评估的合理条件，但笔记本真实峰值显存、推理延迟和实机成功率尚未测量。本次仅完成资料/源码/文件核对与方案制定，没有安装环境、加载模型运行推理、操作机械臂或更改远端仓库。

用户已确认：G:\LeRobot\.venv 和 source 仍在；相机位置、焦距和校准保持 0918 状态。EgoSMPLX/WiLoR 任务保持暂停。

核对仓库：https://github.com/cdh290718-oss/LeRobot
固定提交：da4bdcf422d97f8e5332bbae0875c5a6487ecd4d。

**环境选择。**

| 项目 | 采用方案 | 理由 |
| --- | --- | --- |
| 系统 | Windows 原生 PowerShell | 已有 LeLab、双 UVC 相机和 COM 串口工作记录；本阶段无需迁移 WSL/Docker |
| Python | G:\LeRobot\.venv\Scripts\python.exe；预期3.12.14 | 以专用环境实际检查为准，不用当前系统 PATH 中的3.14代替 |
| PyTorch | 优先保留已验证的2.10.0+cu128 | 仓库验收记录对应此版本 |
| torchvision / torchcodec | 0.25.0+cu128 / 0.10.0 | 保留同一次验收通过的配套依赖 |
| LeRobot | 现有 source/lerobot-0.6.0 editable源码 | 含项目版本与本地修改，不直接安装上游最新main覆盖 |
| Transformers | 5.5.4 | 已验收版本，落在源码>=5.4,<5.6范围 |
| 视频读取 | 先沿用已通过编解码验收的配置；离线读取可明确选PyAV | 不复制A100开发机的cu118整套requirements |
| 显卡驱动 | 保留用户报告的582.05 | CUDA驱动显示13.0不要求PyTorch runtime也为13.0；cu128可在足够新的驱动上运行 |

你报告的 torch2.11.0+cu128 本身满足该源码 torch>=2.7,<2.12 范围，并非认定这个torch版本不能用。但 Python3.14 不适合作为本项目第一条部署路线：源码要求 numpy>=2.0,<2.3，NumPy2.2.6 官方支持 Python3.10–3.13。新建或强行升级依赖会偏离已验证环境。若现有专用环境已变化，应先导出清单，再决定是否在新环境复原，不直接降级系统环境。

仅核对环境的 PowerShell 示例（不连接机械臂、不做模型推理）：

```powershell
$lerobotPython = 'G:\LeRobot\.venv\Scripts\python.exe'
& $lerobotPython -c "import sys,torch,torchvision,importlib.metadata as m; print(sys.executable); print(sys.version); print('torch',torch.__version__,'CUDA runtime',torch.version.cuda); print('torchvision',torchvision.__version__); print('lerobot',m.version('lerobot')); print('transformers',m.version('transformers')); print('cuda_available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
& $lerobotPython -m pip check
```

**模型与必要资源。**

开发机已只读确认下列两个真实目录均存在，各含7个文件：

```text
/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/outputs/train10k/checkpoints/006000/pretrained_model/
/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/outputs/train10k/checkpoints/010000/pretrained_model/
```

建议本地分别放在新的独立目录：

```text
G:\LeRobot\models\yellow_cube_0918_v1\006000\pretrained_model\
G:\LeRobot\models\yellow_cube_0918_v1\010000\pretrained_model\
```

每个目录必须完整迁移：

```text
model.safetensors
config.json
policy_preprocessor.json
policy_preprocessor_step_5_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
```

主权重每份约906.7MB（约865MiB），两份顺序评估，不同时常驻GPU。推理不需要training_state优化器文件。GitHub的checkpoint目录只有JSON，不是完整模型包。迁移应记录远端和本地SHA256，使用实际Windows路径加载；config内历史Linux pretrained_path不应直接当作本地入口。

基础资源也要准备：当前 SmolVLMWithExpertModel 构造器中 load_vlm_weights=true 会调用 AutoModelForImageTextToText.from_pretrained，随后还调用AutoProcessor.from_pretrained。当前模型ID为 HuggingFaceTB/SmolVLM2-500M-Video-Instruct，tokenizer同名。应先核对 G:\LeRobot\cache\huggingface 的对应资源是否完整；允许联网时准备缓存，再做断网加载检查。仅复制微调权重不足以保证离线加载。第一次部署保留现有加载语义，不未经验证改成随机初始化或修改load_vlm_weights。

6k历史验证loss0.0763，10k为0.0915，样本有限且损失有随机性，不能直接据此认定6k实机更好。方案保留两个候选，以同一批输入和评估协议比较。

**观察、动作和相机保持训练语义。**

模型任务文本：

```text
Pick up the yellow cube and place it into the mesh container.
```

| 物理视角 | 原逻辑相机名 | 归档OpenCV索引 | 模型输入 |
| --- | --- | ---: | --- |
| 外部相机 | 3 | 1 | observation.images.camera1 |
| 腕部相机 | 1 | 2 | observation.images.camera2 |

索引只是归档记录；接入前用实际画面确认，USB插拔可能重排。保持两路640×480、30fps、MJPG采集。图像张量变换、resize/pad及归一化服从checkpoint配置，不能因config中出现256或512就手工改变相机采集分辨率。配置列有camera3，但当前prepare_images读取实际存在的图像，empty_cameras=0，本项目仍只用两路，不复制一张图伪造第三视角。

实际dataset确认state/action均6维、顺序相同：shoulder_pan.pos、shoulder_lift.pos、elbow_flex.pos、wrist_flex.pos、wrist_roll.pos、gripper.pos。沿用原校准和关节单位，经保存的processor归一化/反归一化，不把输出当作XYZ位姿。

**已发现的入口兼容问题：部署前需显式处理。**

固定提交中的 `source/lerobot-0.6.0/src/lerobot/rollout/context.py` 在视觉特征校验中使用cfg.rename_map；构建processor时又明确传入：

```python
"rename_observations_processor": {"rename_map": cfg.rename_map}
```

`RolloutConfig.rename_map` 默认是空字典。现有LeLab `lelab/rollout.py` 的启动命令没有传--rename_map。因而即使checkpoint已经保存训练时的映射，直接使用原界面启动仍会遇到逻辑名1/3与camera1/camera2不一致的检查，且空配置会覆盖保存映射。这是源码层面确认，尚未在Windows实跑复现。

首轮部署优先用可审计的命令行配置显式传以下映射；CLI跑通后，再使LeLab入口读取并传递同一映射。保持采集配置不变。

```json
{"observation.images.3":"observation.images.camera1","observation.images.1":"observation.images.camera2"}
```

当前实际实机入口是 `python -m lerobot.scripts.lerobot_rollout`；`lerobot-eval`不应直接当作本项目SO101实机入口。本次未修改这些文件，也未调用rollout。

**显存、精度与延迟方案。**

当前8151MiB中占用2509MiB，已有约5642MiB空闲；用户预计的5.5–7GB应理解为可能可用的总量，不是还能额外释放5.5–7GB。以实际测量为准。

先采用 batch=1、单模型、eval/inference_mode，保留num_steps=10、chunk_size=50、n_action_steps=50、图像预处理和use_amp=true；torch.compile=false；不开量化、TensorRT、额外推理服务或多模型并发。先关闭非必要可视化/后台应用来减少干扰，不修改驱动或系统全局GPU设置。

精度需做一个明确核对：远端训练使用BF16；现有同步推理代码为torch.autocast(device_type='cuda')，未指定dtype，CUDA默认通常为FP16，不能把use_amp=true写成“已经保证BF16”。离线先检查现有入口对应AMP行为的有限值和输出，再比较显式BF16基线；若决定保持BF16，应在推理上下文中明确设置dtype并记录改动，而非单纯对整个模型调用half()。

仓库训练记录的A100峰值分配2.081GiB能支持“8GB值得做本地推理测试”的判断，但不能直接作为RTX5060显存实测值，磁盘权重大小也不能等同运行峰值。用torch.cuda.max_memory_allocated/reserved及nvidia-smi同时记录，覆盖加载、预热、稳定推理和视频记录。

延迟要单独测真实生成action chunk的调用。select_action在队列非空时只弹出缓存动作，连续调用的平均耗时可能严重低估网络计算；使用predict_action_chunk或每次清空队列的对应基准，并在计时边界同步CUDA。建议预热后记录至少100次的P50/P95、峰值显存，以及完整预处理/后处理耗时。

50步动作在30Hz下覆盖约1.67秒；这不等于网络达到30次推理/秒，也不意味着可以长期忽略新观察。先验证sync基线是否产生周期停顿，再根据实测决定是否测试RTC及执行窗口。初始不随意降低整个控制频率或截短chunk来冒充加速，避免同时改变训练时间尺度与控制行为。

**执行顺序与验收。**

1. 核查G盘解释器、依赖、源码路径和模型/基础资源文件；记录版本清单及checkpoint哈希。
2. 用0918原验证段的记录图像和state做纯离线加载与推理，机械臂不连接。验证图像映射、6维动作、数值有限、processor一致和反归一化后尺度；同一输入/种子比较6k与10k。
3. 测显存、加载时间、真实chunk延迟，分别报告FP16/BF16行为和冷启动/稳态表现。Windows数据加载先用num_workers=0。
4. 双相机和串口进行只读观测检查，确认实际端口与画面对应；实时影像/state送模型只记录预测，执行器不接收动作。
5. 完成上述验收后再进行有限时长的实机试验，保持原校准，有明确停止途径；记录是否抓起、是否落入容器、耗时及失败原因。开始前处理相机rename_map入口问题；实机命令执行会运动机械臂，不能当作单纯的模型benchmark。
6. 用同一组事先固定的位置和次数比较候选模型（可先每个10次，报告原始成功次数）；训练/验证loss不替代抓取成功率。

当前缺少的是本机推理测量，而不是新的训练权重。新方案不要求重新训练、重新校准或安装WSL。现有采集与前端本地修改继续保留；如后续记录评估视频，沿用用户要求的streaming encoding，避免复用旧脚本覆盖现有数据。

**可复核来源。**

- 仓库交接与环境：https://github.com/cdh290718-oss/LeRobot/blob/da4bdcf422d97f8e5332bbae0875c5a6487ecd4d/docs/交接文档.md
- 现有环境验收：https://github.com/cdh290718-oss/LeRobot/blob/da4bdcf422d97f8e5332bbae0875c5a6487ecd4d/local/environment-records/verification.json
- 依赖约束：https://github.com/cdh290718-oss/LeRobot/blob/da4bdcf422d97f8e5332bbae0875c5a6487ecd4d/source/lerobot-0.6.0/pyproject.toml
- 实机处理器覆盖：https://github.com/cdh290718-oss/LeRobot/blob/da4bdcf422d97f8e5332bbae0875c5a6487ecd4d/source/lerobot-0.6.0/src/lerobot/rollout/context.py
- LeLab启动命令：https://github.com/cdh290718-oss/LeRobot/blob/da4bdcf422d97f8e5332bbae0875c5a6487ecd4d/source/lelab-9a182993a2a9a3b04312c13ff8365f68d379fe80/lelab/rollout.py
- NumPy2.2.6支持范围：https://numpy.org/doc/2.3/release/2.2.6-notes.html
- NVIDIA驱动向后兼容：https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- PyTorch版本配套：https://pytorch.org/get-started/previous-versions/
