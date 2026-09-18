from pathlib import Path
from datetime import datetime
import ast

path = Path(r'G:\LeRobot\source\lelab-9a182993a2a9a3b04312c13ff8365f68d379fe80\lelab\record.py')
source = path.read_text(encoding='utf-8')
old = '''                request.dataset_repo_id = re.sub(r"[^A-Za-z0-9._-]", "_", request.dataset_repo_id)
'''
new = old + '''                # LeRobot requires namespace/name even for local-only recording.
                if request.push_to_hub:
                    raise ValueError("Uploading requires a dataset ID in username/dataset format.")
                request.dataset_repo_id = f"local/{request.dataset_repo_id}"
'''
assert source.count(old) == 1, 'Expected a unique normalization block'
updated = source.replace(old, new, 1)
ast.parse(updated)
backup = Path(r'G:\LeRobot\logs') / ('record-before-local-namespace-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.py.bak')
backup.write_bytes(path.read_bytes())
path.write_text(updated, encoding='utf-8')
print('Patched:', path)
print('Backup:', backup)
