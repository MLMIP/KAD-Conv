"""Prepare the official DIOR HBB layout without modifying the downloaded dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import yaml

NAMES = [
    'airplane',
    'airport',
    'baseballfield',
    'basketballcourt',
    'bridge',
    'chimney',
    'dam',
    'Expressway-Service-area',
    'Expressway-toll-station',
    'golffield',
    'groundtrackfield',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'storagetank',
    'tenniscourt',
    'trainstation',
    'vehicle',
    'windmill',
]


def canonical(value):
    return ''.join(x for x in value.lower() if x.isalnum())


CLASS = {canonical(v): i for i, v in enumerate(NAMES)}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def convert(xml):
    node = ET.parse(xml).getroot()
    width, height = float(node.findtext('size/width')), float(node.findtext('size/height'))
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid image size: {xml}')
    rows = []
    for obj in node.findall('object'):
        name = canonical(obj.findtext('name') or '')
        if name not in CLASS:
            raise ValueError(f'Unknown category: {xml}: {name}')
        b = obj.find('bndbox')
        xmin, ymin, xmax, ymax = (float(b.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax'])
        # DIOR's existing labels use original XML coordinates, with no -1 shift.
        values = [
            ((xmin + xmax) / 2) * (1 / width),
            ((ymin + ymax) / 2) * (1 / height),
            (xmax - xmin) * (1 / width),
            (ymax - ymin) * (1 / height),
        ]
        rows.append((CLASS[name], *values))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--yaml', type=Path, default=Path('outputs/data/DIOR.yaml'))
    ap.add_argument('--example-id', default='12500')
    ap.add_argument('--existing-labels', type=Path, help='Optional source-label directory for semantic audit')
    args = ap.parse_args()
    raw, out = args.raw.resolve(), args.output.resolve()
    if out == raw or out.is_relative_to(raw):
        raise ValueError('Output must be outside the raw dataset')
    out.mkdir(parents=True, exist_ok=True)
    annotation = raw / 'Annotations'
    if (annotation / 'Horizontal Bounding Boxes').is_dir():
        annotation = annotation / 'Horizontal Bounding Boxes'
    report = {
        'raw_root': str(raw),
        'output': str(out),
        'classes': NAMES,
        'splits': {},
        'label_lines': {},
        'reference_mismatches': [],
        'reference_max_abs': 0.0,
        'source_split_sha256': {},
    }
    members = {}
    for split, expected in [('train', 5862), ('val', 5863), ('test', 11738)]:
        source = raw / 'ImageSets/Main' / f'{split}.txt'
        ids = [x.strip() for x in source.read_text().splitlines() if x.strip()]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f'{split} split count: {len(ids)}, expected {expected}')
        members[split] = ids
        report['source_split_sha256'][split] = sha(source)
        imgdir = raw / ('JPEGImages-test' if split == 'test' else 'JPEGImages-trainval')
        total = 0
        for index, stem in enumerate(ids):
            image = imgdir / (stem + '.jpg')
            if not image.is_file():
                raise FileNotFoundError(image)
            dest = out / 'images' / split / image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(image)
            rows = convert(annotation / (stem + '.xml'))
            total += len(rows)
            label = out / 'labels' / split / (stem + '.txt')
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(''.join(' '.join(str(v) for v in row) + '\n' for row in rows))
            if args.existing_labels and split == 'test':
                reference = args.existing_labels / (stem + '.txt')
                if not reference.exists():
                    reference = args.existing_labels / split / (stem + '.txt')
                old = [tuple(float(v) for v in l.split()) for l in reference.read_text().splitlines() if l.strip()]
                if len(old) != len(rows):
                    report['reference_mismatches'].append(
                        {'id': stem, 'reason': 'count', 'old': len(old), 'new': len(rows)}
                    )
                else:
                    delta = max(
                        (abs(float(a) - float(b)) for x, y in zip(old, rows) for a, b in zip(x, y)), default=0.0
                    )
                    report['reference_max_abs'] = max(report['reference_max_abs'], delta)
                    if delta > 1e-9:
                        report['reference_mismatches'].append({'id': stem, 'reason': 'values', 'max_abs': delta})
            if index % 1000 == 0:
                print(f'PREP_PROGRESS split={split} images={index+1}/{len(ids)}', flush=True)
        report['splits'][split] = len(ids)
        report['label_lines'][split] = total
    assert not set(members['train']) & set(members['val']) and not set(members['train'] + members['val']) & set(
        members['test']
    )
    (out / 'splits').mkdir(exist_ok=True)
    split_by_id = {stem: split for split, ids in members.items() for stem in ids}
    for split, ids in [('trainval', members['train'] + members['val']), ('test', members['test'])]:
        images = [str(out / 'images' / split_by_id[s] / (s + '.jpg')) for s in ids]
        (out / 'splits' / f'{split}.txt').write_text('\n'.join(images) + '\n')
    cfg = {
        'path': str(out),
        'train': 'splits/trainval.txt',
        'val': 'images/test',
        'test': 'images/test',
        'nc': 20,
        'names': NAMES,
    }
    args.yaml.parent.mkdir(parents=True, exist_ok=True)
    args.yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if args.example_id not in members['test']:
        raise ValueError('Demo ID must be in official test split')
    (out / 'examples').mkdir(exist_ok=True)
    shutil.copyfile(out / 'images/test' / f'{args.example_id}.jpg', out / 'examples' / f'dior_{args.example_id}.jpg')
    report['yaml'] = str(args.yaml.resolve())
    report['example_image'] = str(out / 'examples' / f'dior_{args.example_id}.jpg')
    (out / 'preparation_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report['reference_mismatches']:
        raise SystemExit('Generated labels differ from the source experiment; see manifest')


if __name__ == '__main__':
    main()
