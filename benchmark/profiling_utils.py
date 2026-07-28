"""
Benchmark profiling utility - shared across all projects.
Measures: model_size (params + MB), inference_speed (seconds), memory_used (peak GPU MB).
"""
import time
import json
import os
import torch
import torch.nn as nn
from datetime import datetime


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model):
    """Get model size in MB (assuming float32)."""
    total_params, _ = count_parameters(model)
    # Each param is 4 bytes (float32) or 2 bytes (float16)
    # Check actual dtype of first parameter
    try:
        first_param = next(model.parameters())
        bytes_per_param = first_param.element_size()
    except StopIteration:
        bytes_per_param = 4
    return (total_params * bytes_per_param) / (1024 * 1024)


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def get_peak_gpu_memory_mb():
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


class BenchmarkProfiler:
    """Context manager for benchmarking inference."""

    def __init__(self, project_name, model=None, log_dir=None):
        self.project_name = project_name
        self.model = model
        self.log_dir = log_dir or f"/home/yanghaotian/server_data/yanghaotian/benchmark_results/{project_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.results = {
            "project": project_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_size": "256x176",
            "denoising_steps": 50,
        }

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        reset_gpu_memory_stats()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.results["inference_time_seconds"] = round(time.time() - self.start_time, 4)
        self.results["peak_gpu_memory_mb"] = round(get_peak_gpu_memory_mb(), 2)

    def record_model_info(self, model, model_name="model"):
        """Record model size information."""
        total_params, trainable_params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        self.results.setdefault("models", {})[model_name] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "size_mb": round(size_mb, 2),
        }

    def record_all_models(self, models_dict):
        """Record info for multiple models. models_dict = {name: model}"""
        for name, model in models_dict.items():
            if model is not None:
                self.record_model_info(model, name)

    # --- Alternative API for manual logging (CFLD-style) ---

    def log_model_size(self, name, num_params, size_mb):
        """Manually log model size (alternative to record_model_info)."""
        self.results.setdefault("models", {})[name] = {
            "total_parameters": num_params,
            "trainable_parameters": num_params,
            "size_mb": round(size_mb, 2),
        }

    def log_inference_speed(self, time_seconds):
        """Manually log inference speed."""
        self.results["inference_time_seconds"] = round(time_seconds, 4)

    def log_memory_used(self, mem_mb):
        """Manually log peak GPU memory."""
        self.results["peak_gpu_memory_mb"] = round(mem_mb, 2)

    def save(self):
        """Alias for save_log()."""
        return self.save_log()

    def save_log(self):
        """Save benchmark results to JSON log file."""
        # Calculate total model size
        total_size = 0
        total_params = 0
        if "models" in self.results:
            for m in self.results["models"].values():
                total_size += m["size_mb"]
                total_params += m["total_parameters"]
        self.results["total_model_size_mb"] = round(total_size, 2)
        self.results["total_parameters"] = total_params

        log_file = os.path.join(self.log_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"Benchmark results saved to: {log_file}")
        print(f"{'='*60}")
        print(json.dumps(self.results, indent=2, ensure_ascii=False))
        return log_file
