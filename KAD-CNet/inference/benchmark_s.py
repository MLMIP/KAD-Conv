import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.runtime import load_model, StaticGraph
import torch


def measure(model, x, warmup, iterations, trials):
    with torch.no_grad():
        graph = StaticGraph(model, x)
    for _ in range(warmup):
        graph(x, clone=False)
    torch.cuda.synchronize()
    values = []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.graph.replay()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end) / iterations)
    median = statistics.median(values)
    return {'trial_latency_ms': values, 'median_latency_ms': median, 'model_fps': 1000 / median}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='weights/kad_cnet_s_dior_mgl.pt')
    p.add_argument('--config')
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=80)
    p.add_argument('--iterations', type=int, default=400)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--output', type=Path, default=Path('outputs/benchmark.json'))
    p.add_argument('--checkpoint-key', choices=['auto', 'ema', 'model'], default='auto')
    a = p.parse_args()
    if a.batch_size != 1:
        raise ValueError('The S acceleration profile uses batch1')
    torch.manual_seed(0)
    x = torch.randn(1, 3, a.imgsz, a.imgsz, device='cuda')
    raw = load_model(a.weights, 'raw', checkpoint_key=a.checkpoint_key)
    fast = load_model(a.weights, 'optimized', config=a.config, checkpoint_key=a.checkpoint_key)
    report = {
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'dtype': 'strict_fp32',
        'tf32': False,
        'batch': 1,
        'image_size': a.imgsz,
        'protocol': 'both raw and optimized use eval, Conv-BN fusion and CUDA Graph; model-forward only; excludes compilation/capture/warmup/preprocess/copy/NMS/I-O',
        'warmup': a.warmup,
        'iterations': a.iterations,
        'trials': a.trials,
        'raw': measure(raw, x, a.warmup, a.iterations, a.trials),
        'optimized': measure(fast, x, a.warmup, a.iterations, a.trials),
    }
    report['speedup'] = report['raw']['median_latency_ms'] / report['optimized']['median_latency_ms']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
