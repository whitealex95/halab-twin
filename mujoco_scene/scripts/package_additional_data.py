"""Package the separate reference photographs without changing the raw capture archive."""
from pathlib import Path
import hashlib, zipfile
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'additional_data'
OUT = ROOT / 'artifacts'
OUT.mkdir(exist_ok=True)
files = sorted(SOURCE.glob('*.jpg'))
assert len(files) == 3
checks = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(SOURCE / 'SHA256SUMS.txt').write_text(''.join(f'{sha}  {name}\n' for name, sha in checks.items()))
archive = OUT / 'halab-additional-data.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(SOURCE.iterdir()):
        if p.is_file(): z.write(p, 'additional_data/' + p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name, sha in checks.items():
        assert hashlib.sha256(z.read('additional_data/' + name)).hexdigest() == sha
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
(OUT / 'halab-additional-data.sha256').write_text(f'{sha}  {archive.name}\n')
print(f'{archive.name}: {len(files)} original photos, {archive.stat().st_size} bytes, SHA-256 {sha}')
