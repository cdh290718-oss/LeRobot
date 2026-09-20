"""Recompute the returned Windows metrics without running inference or hardware."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
from safetensors.torch import load_file

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PROJECT = ROOT.parent.parent
ARCHIVE = PROJECT/'cui_local_computer_files/results/20260919_113316_962_Run_return.zip'
TRAIN = Path('/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918/outputs/train10k/checkpoints')
manifest = json.loads((ROOT/'offline_kit/samples/manifest.json').read_text())
expected = json.loads((ROOT/'offline_kit/checkpoint_manifest.json').read_text())
stats = load_file(str(TRAIN/'010000/pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
std = stats[next(k for k in stats if 'action' in k and k.endswith('std'))].numpy().reshape(6).astype(np.float64)
results = {'source_zip': str(ARCHIVE), 'source_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
           'runs': {}, 'comparisons': {}, 'limits': ['33 held-out frames from 11 episodes, not a full validation set.',
               '100 latency calls repeat the same 33 inputs; not 100 distinct validation samples.',
               'Raw per-call timing and GPU clock/power telemetry are absent; cannot identify latency fluctuation cause.',
               'No camera, serial, video recording, action execution or grasp-success measurements.']}
arrays = {}
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    report = json.loads(z.read('summary.json'))
    assert report['status'] == 'passed' and not report['robot_connected']
    results['environment'] = report['environment']
    for step, ck in report['assets']['checkpoints'].items():
        assert ck['ok']
        for name, ref in expected['checkpoints'][step]['files'].items():
            assert ck['files'][name]['sha256'] == ref['sha256']
    for run in report['runs']:
        key = run['folder']
        with np.load(io.BytesIO(z.read(key+'/predictions.npz')), allow_pickle=False) as n:
            pred = n['predicted'].copy(); target = n['reference'].copy(); valid = n['valid'].copy()
            assert np.array_equal(n['sample_indices'], np.arange(33))
        assert pred.shape == target.shape == (33,50,6) and valid.shape == (33,50)
        assert np.isfinite(pred).all() and np.isfinite(target).all()
        normalized_error = (pred.astype(np.float64) - target.astype(np.float64)) / (std + 1e-8)
        mse = float(np.mean(normalized_error[valid]**2))
        np.testing.assert_allclose(mse, run['held_out_normalized_action_mse'], rtol=1e-5)
        mae = np.abs(pred.astype(np.float64)-target.astype(np.float64))[valid].mean(0)
        np.testing.assert_allclose(mae, list(run['held_out_mae_per_joint'].values()), rtol=1e-5)
        episode_error = {}
        for ep in manifest['held_out_episodes']:
            selected = np.array([s['episode_index'] == ep for s in manifest['samples']])
            episode_error[str(ep)] = float(np.mean(normalized_error[selected][valid[selected]]**2))
        with np.load(ROOT/'developer_validation'/key/'predictions.npz', allow_pickle=False) as dev:
            assert np.array_equal(target, dev['reference']) and np.array_equal(valid, dev['valid'])
            diff = np.abs(pred.astype(np.float64)-dev['predicted'].astype(np.float64))
        log = z.read(key+'/console.log').decode('utf-8', errors='replace')
        assert 'Traceback (most recent call last)' not in log
        results['runs'][key] = {'status': run['status'], 'recomputed_normalized_mse': mse,
             'recomputed_mae_per_joint': mae.tolist(), 'episode_mse': episode_error,
             'latency_ms': run['latency_ms'], 'load_seconds': run['load_seconds'],
             'cold_pipeline_ms': run['cold_first_chunk_ms']['pipeline_ms'],
             'steady_memory': run['steady_memory'], 'nvidia_smi_after': run['nvidia_smi_after'],
             'windows_vs_developer_mae_per_joint': diff.mean((0,1)).tolist(),
             'windows_vs_developer_max_abs': float(diff.max()),
             'saved_unnormalizer_probe_passed': run['saved_unnormalizer_probe_passed']}
        arrays[key] = pred
    for amp in ('fp16','bf16'):
        a = results['runs']['006000_'+amp]; b = results['runs']['010000_'+amp]
        results['comparisons'][amp] = {'mse_relative_reduction_10k_vs_6k': 1-b['recomputed_normalized_mse']/a['recomputed_normalized_mse'],
             'episodes_with_lower_10k_error': [ep for ep in a['episode_mse'] if b['episode_mse'][ep] < a['episode_mse'][ep]],
             'total_episodes': 11}
    results['fp16_vs_bf16'] = report['fp16_vs_bf16']
(OUT/'ANALYSIS.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
rows = []
for key,r in results['runs'].items():
    a=r['latency_ms']['chunk_ms']; b=r['latency_ms']['pipeline_ms']; m=r['steady_memory']
    rows.append(f"| {key} | {a['p50']:.1f} / {a['p95']:.1f} | {b['p50']:.1f} / {b['p95']:.1f} | {m['peak_allocated_mib']:.1f} / {m['peak_reserved_mib']:.1f} | {r['recomputed_normalized_mse']:.6f} |")
text = '''# Windows RTX 5060 本地回传报告复核

结论：本地离线部署已跑通。复用 G:\\LeRobot\\.venv 的 Python 3.12.14、torch 2.10.0+cu128 和现有 LeRobot 0.6.0 源码即可继续下一阶段，不需要为本次推理升级环境。此结论仅覆盖本次双图像、batch=1、50步动作序列的离线推理。

已检查 ZIP CRC、两份 checkpoint 的全部 SHA256、预测数组、参考动作一致性、归一化 MSE 与逐关节 MAE。四个组合都覆盖 33 个独立样本、各测量 100 次真实 chunk 调用，另有冷启动和5次预热。输出形状全部正确，未发现 NaN/Inf，保存的反归一化统计量探针通过。

| 组合 | 模型 chunk P50 / P95 (ms) | 含前后处理 P50 / P95 (ms) | PyTorch 峰值分配 / 预留 (MiB) | 生成动作归一化 MSE |
| --- | --- | --- | --- | --- |
''' + '\n'.join(rows) + '''

**如何解释。**

- 10k 的生成动作归一化 MSE 比6k低约20.7%，两种精度得出同方向结果；逐关节平均误差也均下降。它与此前 flow-matching 验证 loss 是不同指标，不能混为一谈，更不代表实机成功率提高20.7%。
- 模型实际由 PyTorch 分配约928–937 MiB，预留约958–1002 MiB。nvidia-smi在四轮末尾的整卡占用约3109–3284 MiB、空闲4527–4702 MiB。这些是不同统计口径；不把PyTorch分配量当作全部进程/桌面显存，也不把末尾快照当作全程峰值。
- 8GB显存对本次配置有余量。双相机采集、实时状态读取、视频记录和控制循环尚未接入，不能由离线结果直接判断整套系统表现。
- 10k FP16的chunk P50为414ms，但P95为929ms；相同架构的6k FP16则为844/897ms。不能把这个差异解释为10k权重在结构上更快。10k BF16的P95较低，但平均值和中位数也有明显差异。现有包没有逐次耗时、频率、功率、温度和电源状态，不能归因于电源策略、散热或后台任务。
- 加载时间约12.7–23.9秒；首次完整pipeline约1.23–2.50秒，正式使用前应预热。
- 一次模型调用生成50步动作；按训练30Hz，覆盖约1.67秒。0.4–0.9秒级序列生成耗时不等于模型能每33ms重新观察和决策。同步入口在生成新chunk时可能停顿；后续需测真实循环并决定异步/执行窗口，不能简单把50除以推理耗时当作闭环帧率。

**下一步选择。**

建议以10k作为主要候选。先用10k BF16作为只读实时输入测试基线，保留FP16作为速度对照；这是基于本次P95和误差的暂定选择，不认定BF16始终更快。接入前仍需显式保留 observation.images.3 → camera1、observation.images.1 → camera2 映射，并确保未来入口明确指定BF16，而非仅设置use_amp=true。

下一阶段先核对两路实际画面与状态输入、只保存预测；在同一电源/性能设置下补记逐次延迟与GPU遥测。完成输入与控制时序验收后，再开展有限时长的实机评估。这次没有连接或控制机械臂。

**非阻塞提示。**

环境没有pip，因此pip check并未完成；这不构成依赖全通过的证据，也没有阻止实际模型加载和推理。日志中只有torch_dtype弃用提示，未发现异常堆栈。无需仅为这两项重建当前能运行的环境。

本地与开发机的参考动作逐值相同；预测接近但不逐位一致。详细跨平台差异、每个验证episode的误差和所有复核值见同目录ANALYSIS.json。不同平台和精度的数值差异不能直接当作关节角度误差；这里的动作采用原数据集与校准定义的单位。
'''
(OUT/'REVIEW_zh.md').write_text(text,encoding='utf-8')
print(json.dumps(results['comparisons'],ensure_ascii=False,indent=2))
print(OUT/'REVIEW_zh.md')
