import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime_env import configure

configure()


def numerical_protocol():
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 10


def acceleration_config(config=None):
    profile = Path(config) if config else ROOT / 'inference/config/dior_s_a100.json'
    for key, value in json.loads(profile.read_text()).items():
        os.environ[key] = str(value)
    os.environ['KADC_DCN16_SOURCE_DIR'] = str(ROOT / 'inference/acceleration')
    sys.path.insert(0, str(ROOT / 'inference/acceleration'))


def load_model(weights, backend='raw', device='cuda:0', config=None, fuse=True, checkpoint_key='auto'):
    if backend not in {'raw', 'optimized', 'graph'}:
        raise ValueError(backend)
    if backend != 'raw':
        if config is None:
            records = json.loads((ROOT / 'weights/manifest.json').read_text())['checkpoints']
            record = next((r for r in records if Path(r['file']).name == Path(weights).name), None)
            if record is not None:
                config = ROOT / record['acceleration_config']
        acceleration_config(config)

    import torch

    numerical_protocol()
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    selected_key = checkpoint_key
    if selected_key == 'auto':
        manifest_path = ROOT / 'weights/manifest.json'
        records = json.loads(manifest_path.read_text())['checkpoints']
        record = next((r for r in records if Path(r['file']).name == Path(weights).name), None)
        selected_key = record['checkpoint_key'] if record else ('ema' if checkpoint.get('ema') is not None else 'model')
    if selected_key not in ('model', 'ema'):
        raise ValueError('Select model or ema checkpoint entry')
    model = checkpoint.get(selected_key)

    if model is None:
        raise ValueError('Checkpoint lacks model/ema')
    model = model.float().eval()
    if fuse:
        model = model.fuse()
    model = model.to(device).eval()
    if backend != 'raw':
        if not str(device).startswith('cuda'):
            raise ValueError('Acceleration requires CUDA')
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise ValueError('The supplied acceleration profile is tested for A100 (SM80); use raw on other hardware')
        from kadc_fused_nsmall import patch
        from kadc_dcn16_backend import install

        model = patch(model)
        install(model)
    import models.yolo, models.module.KADC.module as module

    for path in [models.yolo.__file__, module.__file__]:
        if not Path(path).resolve().is_relative_to(ROOT):
            raise RuntimeError(f'models imported from outside this repository: {path}')
    model._release_checkpoint_key = selected_key
    return model


def leaves(value):
    import torch

    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in leaves(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in leaves(v)]
    return []


def clone_output(value):
    import torch

    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [clone_output(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_output(v) for v in value)
    if isinstance(value, dict):
        return {k: clone_output(v) for k, v in value.items()}
    return value


class StaticGraph:
    """Own each capture's output and clone frame results before the next replay."""

    def __init__(self, forward, sample):
        import torch

        self.input = torch.empty_like(sample)
        self.input.copy_(sample)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(20):
                forward(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph, stream=stream):
            self.output = forward(self.input)
        torch.cuda.synchronize()

    def __call__(self, value, clone=True):
        import torch

        self.input.copy_(value)
        self.graph.replay()
        return clone_output(self.output) if clone else self.output


def graph_forward(model):
    """Evaluation adapter with a distinct capture for each input shape."""
    original = model.forward
    captures = {}

    def forward(value, augment=False, **kwargs):
        if augment:
            raise ValueError('Graph evaluation does not use test-time augmentation')
        key = (tuple(value.shape), value.dtype, value.device)
        if key not in captures:
            captures[key] = StaticGraph(lambda x: original(x), value)
        return captures[key](value)

    model.forward = forward
    return model
