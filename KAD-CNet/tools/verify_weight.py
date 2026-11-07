"""Check downloaded checkpoints against the sizes and SHA-256 values in weights/manifest.json."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, help='checkpoint to check (default: all listed in the manifest)')
    args = parser.parse_args()

    records = json.loads((ROOT / 'weights/manifest.json').read_text(encoding='utf-8'))['checkpoints']
    if args.weights:
        records = [r for r in records if Path(r['file']).name == args.weights.name]
        if not records:
            raise SystemExit(f'{args.weights.name} is not listed in weights/manifest.json')

    ok = True
    for r in records:
        path = args.weights if args.weights else ROOT / r['file']
        if not path.exists():
            print(f'missing   {r["file"]}')
            ok = False
            continue
        good = path.stat().st_size == r['bytes'] and sha256(path) == r['sha256']
        print(f'{"ok" if good else "MISMATCH":9} {r["file"]}')
        ok &= good
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
