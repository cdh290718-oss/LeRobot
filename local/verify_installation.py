import json
import importlib.metadata as metadata
from pathlib import Path
import torch
import av
import numpy as np

report = {"packages": {p: metadata.version(p) for p in ["lelab", "lerobot", "torch", "torchvision", "datasets", "av", "transformers"]}}
report["cuda_available"] = torch.cuda.is_available()
assert report["cuda_available"]
report["gpu"] = torch.cuda.get_device_name(0)
a = torch.randn(128, 128, device="cuda")
assert torch.isfinite(a @ a).all().item()
report["gpu_operation"] = "passed"
del a
torch.cuda.empty_cache()
import lelab.server
from lelab.utils import config
from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_CALIBRATION
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
report["server_import"] = "passed"
report["smolvla_import"] = "passed"
report["data_root"] = str(HF_LEROBOT_HOME)
report["calibration_root"] = str(HF_LEROBOT_CALIBRATION)
report["robot_config_root"] = config.ROBOTS_PATH
assert Path(config.ROBOTS_PATH).is_relative_to(Path("G:/LeRobot"))
video = Path("G:/LeRobot/tmp/installation-video-check.mp4")
with av.open(str(video), "w") as container:
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    for i in range(10):
        frame = av.VideoFrame.from_ndarray(np.full((48,64,3), i*20, dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame): container.mux(packet)
    for packet in stream.encode(): container.mux(packet)
with av.open(str(video)) as container:
    count = sum(1 for _ in container.decode(video=0))
assert count == 10
report["video_encode_decode_frames"] = count
Path("G:/LeRobot/logs/verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
