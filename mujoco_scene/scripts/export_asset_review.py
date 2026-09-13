"""Export comparison sheets for the manually reviewed existing assets."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
manifest = json.loads((WEB / 'data/manifest.json').read_text())
review = json.loads((ROOT / 'asset_review.json').read_text())
assert {a['name'] for a in review['assets']} == {a['name'] for a in manifest['assets']}
# Representative views spanning the review's source-frame annotations.
for page, indices in enumerate([[9,18,21,33,66], [75,84,102,105,117], [123,135,168,171,174]], 1):
    sheet = Image.new('RGB', (1024, 404 * len(indices)), 'white')
    labels = ImageDraw.Draw(sheet)
    for row, index in enumerate(indices):
        frame = manifest['frames'][index]
        for column, key in enumerate(['rgb', 'render_rgb']):
            sheet.paste(Image.open(WEB / frame[key]).resize((512,384)), (column * 512, row * 404 + 20))
            label = 'Recorded RGB' if column == 0 else 'MuJoCo'
            labels.text((column * 512 + 5, row * 404 + 3), f'Frame {index+1:03} / {label}', fill='black')
    sheet.save(ROOT / f'reference/asset_review_{page}.jpg', quality=90)
print(f'Exported three comparison sheets; review covers {len(review["assets"])} assets.')
