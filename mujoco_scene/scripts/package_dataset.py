"""Package verified original capture files for the public GitHub release."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[2]
checks=json.loads((ROOT/'mujoco_scene/raw_input_checksums.json').read_text())
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
archive=OUT/'halab-raw-dataset.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,strict_timestamps=False) as z:
 for rel,digest in sorted(checks.items()):
  p=ROOT/rel
  assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,rel
  z.write(p,rel)
 z.writestr('RAW_SHA256SUMS.txt',''.join(f'{digest}  {rel}\n' for rel,digest in sorted(checks.items())))
 z.write(ROOT/'DATASET.md','DATASET.md')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(checks)+2
 for rel,digest in checks.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
(OUT/'halab-raw-dataset.sha256').write_text(f'{sha}  {archive.name}\n')
print(f'{archive.name}: {archive.stat().st_size} bytes, {len(checks)} verified raw files, SHA-256 {sha}')
