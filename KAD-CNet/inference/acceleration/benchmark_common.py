#!/usr/bin/env python3
"""Shared utilities for detector end-to-end latency benchmarks.

The timed region starts after a preprocessed FP32 tensor is resident on the
GPU and ends after detector-specific postprocessing.  Image I/O,
preprocessing, H2D/D2D input copies, metric computation and visualization are
outside the timed region.
"""

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def select_paths(list_file, sample_count):
    paths = [line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]
    if sample_count <= 0 or sample_count >= len(paths):
        return paths, list(range(len(paths)))
    if sample_count == 1:
        return [paths[len(paths) // 2]], [len(paths) // 2]
    indices = []
    for index in range(sample_count):
        value = round(index * (len(paths) - 1) / (sample_count - 1))
        if not indices or value != indices[-1]:
            indices.append(value)
    return [paths[index] for index in indices], indices


def preload_square_uint8(paths, image_size):
    """Preload evenly sampled images so disk stalls cannot affect GPU clocks."""
    values = []
    shapes = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            shapes.append([int(image.height), int(image.width)])
            image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        values.append(torch.from_numpy(np.ascontiguousarray(array)))
    return values, shapes


def tensor_leaves(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(tensor_leaves(item))
        return result
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(tensor_leaves(value[key]))
        return result
    return []


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


class GraphRunner:
    def __init__(self, model, static_input, capture_warmup=20):
        self.model = model
        self.static_input = static_input
        self.stream = torch.cuda.Stream(priority=-1)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.no_grad():
            for _ in range(capture_warmup):
                model(static_input)
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(
            self.graph, stream=self.stream, capture_error_mode="global"
        ):
            self.output = model(static_input)
        torch.cuda.synchronize()

    def prepare(self, gpu_input):
        with torch.cuda.stream(self.stream):
            self.static_input.copy_(gpu_input)
        self.stream.synchronize()

    def replay(self):
        self.graph.replay()
        return self.output


def compare_eager_graph(model, runner, gpu_input):
    runner.prepare(gpu_input)
    with torch.no_grad(), torch.cuda.stream(runner.stream):
        eager = [item.clone() for item in tensor_leaves(model(runner.static_input))]
    runner.stream.synchronize()
    runner.prepare(gpu_input)
    with torch.no_grad(), torch.cuda.stream(runner.stream):
        candidate = tensor_leaves(runner.replay())
    runner.stream.synchronize()
    if len(eager) != len(candidate):
        raise RuntimeError(f"graph leaf mismatch: eager={len(eager)} graph={len(candidate)}")
    maximum = 0.0
    mean_sum = 0.0
    for reference, actual in zip(eager, candidate):
        difference = (reference - actual).abs()
        maximum = max(maximum, float(difference.max().item()))
        mean_sum += float(difference.mean().item())
    return {
        "leaves": len(eager),
        "max_abs": maximum,
        "mean_abs": mean_sum / max(len(eager), 1),
    }


def benchmark_detector(
    runner,
    cpu_uint8_inputs,
    postprocess,
    count_detections,
    warmup,
    trials,
):
    device = runner.static_input.device
    start_event = torch.cuda.Event(enable_timing=True)
    model_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    first = cpu_uint8_inputs[0].to(device=device, dtype=torch.float32).unsqueeze(0).div_(255.0)
    runner.prepare(first)
    with torch.no_grad(), torch.cuda.stream(runner.stream):
        for _ in range(warmup):
            processed = postprocess(runner.replay())
    runner.stream.synchronize()
    del processed, first

    wall_ms = []
    gpu_model_ms = []
    gpu_post_ms = []
    gpu_total_ms = []
    detection_counts = []
    trial_wall_means = []

    for _ in range(trials):
        trial_values = []
        for cpu_value in cpu_uint8_inputs:
            gpu_value = cpu_value.to(device=device, dtype=torch.float32).unsqueeze(0).div_(255.0)
            runner.prepare(gpu_value)
            del gpu_value

            wall_start = time.perf_counter()
            with torch.no_grad(), torch.cuda.stream(runner.stream):
                start_event.record(runner.stream)
                output = runner.replay()
                model_event.record(runner.stream)
                processed = postprocess(output)
                end_event.record(runner.stream)
            end_event.synchronize()
            wall_value = (time.perf_counter() - wall_start) * 1000.0
            model_value = start_event.elapsed_time(model_event)
            total_value = start_event.elapsed_time(end_event)

            wall_ms.append(wall_value)
            trial_values.append(wall_value)
            gpu_model_ms.append(model_value)
            gpu_total_ms.append(total_value)
            gpu_post_ms.append(total_value - model_value)
            detection_counts.append(int(count_detections(processed)))
            del processed
        trial_wall_means.append(statistics.fmean(trial_values))

    mean_wall = statistics.fmean(wall_ms)
    return {
        "samples_per_trial": len(cpu_uint8_inputs),
        "trials": trials,
        "total_timed_images": len(wall_ms),
        "trial_mean_wall_ms": trial_wall_means,
        "mean_latency_ms": mean_wall,
        "fps": 1000.0 / mean_wall,
        "p50_latency_ms": percentile(wall_ms, 0.50),
        "p95_latency_ms": percentile(wall_ms, 0.95),
        "latency_std_ms": statistics.pstdev(wall_ms),
        "mean_gpu_model_ms": statistics.fmean(gpu_model_ms),
        "mean_gpu_postprocess_ms": statistics.fmean(gpu_post_ms),
        "mean_gpu_total_ms": statistics.fmean(gpu_total_ms),
        "mean_detections": statistics.fmean(detection_counts),
        "min_detections": min(detection_counts),
        "max_detections": max(detection_counts),
    }


def emit_result(prefix, payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    print(prefix + encoded, flush=True)
