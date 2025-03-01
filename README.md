# CUDA GPU Benchmark Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg)](https://www.docker.com/)

A production-ready CUDA GPU benchmark tool for measuring and monitoring GPU performance through configurable matrix operations.

## Features

- 🚀 Configurable matrix operations with adjustable computational intensity
- 📊 Real-time GPU monitoring (temperature, memory usage, utilization)
- 📝 Comprehensive logging and performance metrics
- 🔒 Safety features with temperature and memory limits
- 🐳 Docker support with CUDA
- 📈 Performance results in JSON format
- ⚙️ Configurable via environment variables or command line
- 🔄 Graceful shutdown and resource cleanup

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.8.0 or later
- Python 3.8 or later
- Docker (optional)
- NVIDIA Container Toolkit (for Docker)

## Quick Start

### Using Docker

```bash
# Build the Docker image
docker build -t cuda-benchmark .

# Run with default settings
docker run --gpus all cuda-benchmark

# Run with custom parameters
docker run --gpus all \
    -v $(pwd)/results:/app/results \
    cuda-benchmark --size 8000 --iterations 100 --power 1.5 --monitor
```

## Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--size`, `-s` | Matrix size (N for NxN matrix) | 5000 |
| `--iterations`, `-i` | Number of iterations | 50 |
| `--power`, `-p` | Power level (0.1 to 2.0) | 1.0 |
| `--monitor` | Enable GPU monitoring | False |
| `--output-dir` | Results directory | benchmark_results |
| `--log-interval` | Progress logging interval | 10 |
| `--gpu-temp-limit` | GPU temperature limit (°C) | 85 |
| `--memory-limit` | GPU memory limit (GB) | 8.0 |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_BENCHMARK_SIZE` | Matrix size |
| `CUDA_BENCHMARK_ITERATIONS` | Number of iterations |
| `CUDA_BENCHMARK_POWER` | Power level |
| `CUDA_BENCHMARK_MONITOR` | Enable monitoring |
| `CUDA_BENCHMARK_OUTPUT_DIR` | Output directory |
| `CUDA_BENCHMARK_LOG_INTERVAL` | Log interval |
| `CUDA_BENCHMARK_TEMP_LIMIT` | Temperature limit |
| `CUDA_BENCHMARK_MEMORY_LIMIT` | Memory limit |

## Output Format

Results are saved in JSON format with the following structure:

```json
{
    "timestamp": "2024-02-13T12:00:00",
    "config": {
        "matrix_size": 5000,
        "iterations": 50,
        "power_level": 1.0
    },
    "performance": {
        "total_time": 120.5,
        "average_operation_time": 2.41,
        "std_operation_time": 0.15,
        "operations_per_second": 0.415
    },
    "gpu_metrics": {
        "temperature": [...],
        "memory_used": [...],
        "utilization": [...]
    }
}
```