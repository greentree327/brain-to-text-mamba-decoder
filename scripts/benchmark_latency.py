"""
Benchmark script for latency and memory profiling.

Measures model inference speed and memory consumption.
"""

import torch
import time
import psutil
import os
from typing import Dict, Tuple
from src.models import MambaDecoder, GRUDecoderBaseline


def get_memory_usage() -> float:
    """Get current process memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def get_gpu_memory_usage() -> float:
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def benchmark_model(model: torch.nn.Module,
                   input_shape: Tuple,
                   device: str = 'cuda',
                   num_iterations: int = 100,
                   warmup_iterations: int = 10) -> Dict:
    """
    Benchmark a model for inference latency and memory.
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape (batch, seq_len, features)
        device: Device to use ('cuda' or 'cpu')
        num_iterations: Number of iterations for measurement
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Dict with benchmark metrics
    """
    model.to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(*input_shape, device=device)
    day_idx = torch.tensor([0] * input_shape[0], device=device)
    
    # Warmup
    print(f"Warming up with {warmup_iterations} iterations...")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model(x, day_idx)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    
    # Benchmark
    print(f"Benchmarking with {num_iterations} iterations...")
    times = []
    
    for _ in range(num_iterations):
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            _ = model(x, day_idx)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = times[10:]  # Drop first few for stability
    
    # Memory profiling
    torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
    with torch.no_grad():
        _ = model(x, day_idx)
    
    peak_memory = get_gpu_memory_usage() if device == 'cuda' else get_memory_usage()
    
    # Calculate statistics
    mean_latency = sum(times) / len(times)
    min_latency = min(times)
    max_latency = max(times)
    
    results = {
        'model_name': model.__class__.__name__,
        'input_shape': input_shape,
        'device': device,
        'mean_latency_ms': mean_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'peak_memory_gb': peak_memory,
        'throughput_samples_per_sec': (input_shape[0] / mean_latency) * 1000,
    }
    
    return results


def print_benchmark_results(results: Dict):
    """Pretty print benchmark results."""
    print("\n" + "="*60)
    print(f"Benchmark Results: {results['model_name']}")
    print("="*60)
    print(f"Device: {results['device']}")
    print(f"Input Shape: {results['input_shape']}")
    print(f"\nLatency Metrics:")
    print(f"  Mean: {results['mean_latency_ms']:.2f} ms")
    print(f"  Min: {results['min_latency_ms']:.2f} ms")
    print(f"  Max: {results['max_latency_ms']:.2f} ms")
    print(f"\nMemory:")
    print(f"  Peak Memory: {results['peak_memory_gb']:.3f} GB")
    print(f"\nThroughput:")
    print(f"  {results['throughput_samples_per_sec']:.1f} samples/sec")
    print("="*60 + "\n")


def run_all_benchmarks():
    """Run benchmarks for all models."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test configurations
    configs = [
        {
            'model_class': MambaDecoder,
            'kwargs': {
                'neural_dim': 513,
                'n_units': 256,
                'n_days': 5,
                'n_classes': 40,
                'n_layers': 3,
            },
            'input_shape': (1, 200, 513),  # Single sample, 200 timesteps
        },
        {
            'model_class': GRUDecoderBaseline,
            'kwargs': {
                'neural_dim': 512,
                'n_units': 256,
                'n_days': 5,
                'n_classes': 40,
                'n_layers': 3,
            },
            'input_shape': (1, 200, 512),
        },
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\nCreating {config['model_class'].__name__}...")
        model = config['model_class'](**config['kwargs'])
        
        print(f"Running benchmark for {config['model_class'].__name__}...")
        results = benchmark_model(
            model,
            config['input_shape'],
            device=device,
            num_iterations=50,
            warmup_iterations=5
        )
        
        all_results.append(results)
        print_benchmark_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for results in all_results:
        print(f"{results['model_name']:<20} Mean: {results['mean_latency_ms']:>8.2f} ms | "
              f"Peak Mem: {results['peak_memory_gb']:>6.3f} GB")
    print("="*60)


if __name__ == "__main__":
    run_all_benchmarks()
