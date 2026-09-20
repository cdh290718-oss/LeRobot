"""Build manifests and portable archives without modifying original checkpoints."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parent
KIT = ROOT/'offline_kit'
TRAIN = Path('/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918')
BASE = TRAIN/'cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def manifest():
    info = {'repository': 'https://github.com/cdh290718-oss/LeRobot',
            'repository_commit': 'da4bdcf422d97f8e5332bbae0875c5a6487ecd4d',
            'checkpoints': {}, 'base_cache_files': sorted(p.name for p in BASE.iterdir() if p.is_file())}
    for step in ('006000', '010000'):
        folder = TRAIN/f'outputs/train10k/checkpoints/{step}/pretrained_model'
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(folder.iterdir()) if p.is_file()}
        info['checkpoints'][step] = {'source': str(folder), 'files': files}
    (KIT/'checkpoint_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

def archive():
    files = [p for p in KIT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'results' not in p.parts]
    for full in (False, True):
        path = ROOT/('lerobot_offline_full.zip' if full else 'lerobot_offline_scripts_samples.zip')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
            for f in sorted(files):
                z.write(f, Path('offline_kit')/f.relative_to(KIT))
            if full:
                for step in ('006000', '010000'):
                    folder = TRAIN/f'outputs/train10k/checkpoints/{step}/pretrained_model'
                    for f in sorted(folder.iterdir()):
                        if f.is_file():
                            z.write(f, Path('offline_kit/models')/step/'pretrained_model'/f.name)
        print(path, path.stat().st_size, flush=True)
        path.with_suffix('.zip.sha256').write_text(digest(path)+'  '+path.name+'\n', encoding='utf-8')

if __name__ == '__main__':
    import sys
    manifest()
    if '--archives' in sys.argv:
        archive()
