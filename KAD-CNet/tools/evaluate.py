"""Full DIOR evaluation with explicit input and execution protocols."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.runtime import load_model, graph_forward, numerical_protocol
from runtime_env import configure

configure()
import val as validation
import models.experimental as experimental
import utils.general as general


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='weights/kad_cnet_s_dior_mgl.pt')
    p.add_argument('--data', required=True)
    p.add_argument('--backend', choices=['raw', 'optimized'], default='raw')
    p.add_argument('--protocol', choices=['paper', 'square'], default='paper')
    p.add_argument('--execution', choices=['eager'], default='eager')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--output', type=Path, default=Path('outputs/evaluation'))
    p.add_argument('--checkpoint-key', choices=['auto', 'ema', 'model'], default='auto')
    a = p.parse_args()
    if a.backend == 'optimized' and (a.protocol != 'square' or a.batch_size != 1):
        raise ValueError('Acceleration evaluation uses square512 batch1')
    numerical_protocol()
    general.check_font = lambda *args, **kwargs: None
    original_loader = validation.create_dataloader

    def loader(*args, **kwargs):
        if a.protocol == 'square':
            kwargs.update(rect=False, pad=0.0)
        return original_loader(*args, **kwargs)

    validation.create_dataloader = loader
    selected = {}

    def load(*args, **kwargs):
        model = load_model(a.weights, a.backend, 'cuda:0', checkpoint_key=a.checkpoint_key)
        selected['checkpoint_key'] = model._release_checkpoint_key
        return model

    experimental.attempt_load = load
    sys.argv = [
        'val.py',
        '--data',
        a.data,
        '--weights',
        a.weights,
        '--task',
        'test',
        '--imgsz',
        '512',
        '--batch-size',
        str(a.batch_size),
        '--conf-thres',
        '0.001',
        '--iou-thres',
        '0.4',
        '--max-det',
        '300',
        '--device',
        '',
        '--workers',
        str(a.workers),
        '--project',
        str(a.output.parent),
        '--name',
        a.output.name,
        '--exist-ok',
    ]
    opts = validation.parse_opt()
    validation.opt = opts
    result = validation.run(**vars(opts), plots=False)
    report = {
        'checkpoint_key': selected['checkpoint_key'],
        'weights': a.weights,
        'backend': a.backend,
        'protocol': a.protocol,
        'execution': a.execution,
        'batch_size': a.batch_size,
        'imgsz': 512,
        'conf': 0.001,
        'nms_iou': 0.4,
        'max_det': 300,
        'multi_label': True,
        'tf32': False,
        'metrics': dict(zip(['precision', 'recall', 'mAP50', 'mAP50_95'], [float(v) for v in result[0][:4]])),
        'per_class_mAP50_95': [float(v) for v in result[1]],
        'timings_ms': [float(v) for v in result[2]],
    }
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print('EVALUATION_RESULT=' + json.dumps(report))


if __name__ == '__main__':
    main()
