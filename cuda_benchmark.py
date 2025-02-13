#!/usr/bin/env python3
"""
CUDA GPU Benchmark Tool

This script provides a production-ready GPU benchmark utility using PyTorch CUDA operations.
It supports configurable matrix operations, GPU monitoring, and detailed performance logging.
"""

import os
import sys
import time
import logging
import argparse
import json
from typing import Dict
from pathlib import Path
from datetime import datetime
import threading
import subprocess
from dataclasses import dataclass
import signal

import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration parameters for the CUDA benchmark."""

    matrix_size: int
    iterations: int
    power_level: float
    monitor: bool
    output_dir: Path
    log_interval: int
    gpu_temp_limit: int
    memory_limit_gb: float


class GPUMonitor:
    """
    GPU monitoring class that tracks temperature, memory usage, and utilization.
    """

    def __init__(self, temp_limit: int, memory_limit_gb: float):
        self.temp_limit = temp_limit
        self.memory_limit_gb = memory_limit_gb
        self.should_stop = False
        self._lock = threading.Lock()
        self.metrics: Dict[str, list] = {
            "temperature": [],
            "memory_used": [],
            "utilization": [],
        }

    def start(self) -> None:
        """Start the GPU monitoring thread."""
        self.should_stop = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self) -> None:
        """Stop the GPU monitoring thread."""
        self.should_stop = True
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join()

    def _monitor_loop(self) -> None:
        """Main monitoring loop that collects GPU metrics."""
        while not self.should_stop:
            try:
                # Get GPU statistics using nvidia-smi
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                temp, memory, util = map(float, result.stdout.strip().split(","))

                with self._lock:
                    self.metrics["temperature"].append(temp)
                    self.metrics["memory_used"].append(memory)
                    self.metrics["utilization"].append(util)

                # Check temperature limit
                if temp > self.temp_limit:
                    logger.warning(
                        f"GPU temperature ({temp}°C) exceeded limit ({self.temp_limit}°C)"
                    )
                    self.should_stop = True
                    break

                # Check memory limit
                if memory / 1024 > self.memory_limit_gb:  # Convert MB to GB
                    logger.warning(
                        f"GPU memory usage ({memory / 1024:.2f}GB) exceeded limit ({self.memory_limit_gb}GB)"
                    )
                    self.should_stop = True
                    break

                time.sleep(1)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to get GPU metrics: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in GPU monitoring: {e}")
                break

    def get_metrics(self) -> Dict[str, list]:
        """Return current metrics with thread-safe access."""
        with self._lock:
            return {k: v.copy() for k, v in self.metrics.items()}


class CUDABenchmark:
    """
    Main benchmark class that performs CUDA operations and collects performance metrics.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.gpu_monitor = GPUMonitor(config.gpu_temp_limit, config.memory_limit_gb)
        self.results: Dict = {}

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle termination signals gracefully."""
        logger.info("Received termination signal. Cleaning up...")
        self.gpu_monitor.stop()
        self._cleanup()
        sys.exit(0)

    def _cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        torch.cuda.empty_cache()
        logger.info("Cleaned up GPU resources")

    def _check_gpu_compatibility(self) -> None:
        """Verify GPU compatibility and CUDA availability."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please check GPU and PyTorch installation."
            )

        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda

        logger.info(f"Found {device_count} CUDA device(s)")
        logger.info(f"Using GPU: {device_name}")
        logger.info(f"CUDA Version: {cuda_version}")

    def run_benchmark(self) -> Dict:
        """
        Run the main benchmark operation.

        Returns:
            Dict: Benchmark results and metrics
        """
        try:
            self._check_gpu_compatibility()

            # Start GPU monitoring if enabled
            if self.config.monitor:
                self.gpu_monitor.start()

            # Initialize matrices on GPU
            a = torch.randn(
                self.config.matrix_size, self.config.matrix_size, device="cuda"
            )
            b = torch.randn(
                self.config.matrix_size, self.config.matrix_size, device="cuda"
            )

            # Warm up GPU
            torch.matmul(a, b)
            torch.cuda.synchronize()

            # Main benchmark loop
            start_time = time.time()
            operation_times = []

            for i in range(self.config.iterations):
                if self.gpu_monitor.should_stop:
                    logger.warning("Benchmark stopped due to GPU limits exceeded")
                    break

                op_start = time.time()

                # Matrix multiplication
                c = torch.matmul(a, b)

                # Additional operations based on power level
                for _ in range(int(self.config.power_level * 3)):
                    c = torch.sin(c)
                    c = torch.exp(torch.clamp(c, -1, 1))

                torch.cuda.synchronize()
                operation_times.append(time.time() - op_start)

                if (i + 1) % self.config.log_interval == 0:
                    logger.info(f"Completed iteration {i + 1}/{self.config.iterations}")
                    current_time = time.time()
                    logger.info(
                        f"Time elapsed: {current_time - start_time:.2f} seconds"
                    )

            end_time = time.time()
            total_time = end_time - start_time

            # Collect and save results
            self.results = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "matrix_size": self.config.matrix_size,
                    "iterations": self.config.iterations,
                    "power_level": self.config.power_level,
                },
                "performance": {
                    "total_time": total_time,
                    "average_operation_time": np.mean(operation_times),
                    "std_operation_time": np.std(operation_times),
                    "operations_per_second": len(operation_times) / total_time,
                },
            }

            if self.config.monitor:
                self.gpu_monitor.stop()
                self.results["gpu_metrics"] = self.gpu_monitor.get_metrics()

            # Save results
            self._save_results()

            return self.results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()

    def _save_results(self) -> None:
        """Save benchmark results to output directory."""
        try:
            # Create output directory if it doesn't exist
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            # Save results as JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.config.output_dir / f"benchmark_results_{timestamp}.json"

            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2)

            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CUDA GPU Benchmark Tool")

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=int(os.getenv("CUDA_BENCHMARK_SIZE", "5000")),
        help="Matrix size (N for NxN matrix)",
    )

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=int(os.getenv("CUDA_BENCHMARK_ITERATIONS", "50")),
        help="Number of iterations to run",
    )

    parser.add_argument(
        "-p",
        "--power",
        type=float,
        default=float(os.getenv("CUDA_BENCHMARK_POWER", "1.0")),
        help="Power level (0.1 to 2.0) to adjust computational intensity",
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        default=bool(os.getenv("CUDA_BENCHMARK_MONITOR", False)),
        help="Enable detailed GPU monitoring",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("CUDA_BENCHMARK_OUTPUT_DIR", "benchmark_results")),
        help="Directory to store benchmark results",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=int(os.getenv("CUDA_BENCHMARK_LOG_INTERVAL", "10")),
        help="Interval for logging progress",
    )

    parser.add_argument(
        "--gpu-temp-limit",
        type=int,
        default=int(os.getenv("CUDA_BENCHMARK_TEMP_LIMIT", "85")),
        help="GPU temperature limit in Celsius",
    )

    parser.add_argument(
        "--memory-limit",
        type=float,
        default=float(os.getenv("CUDA_BENCHMARK_MEMORY_LIMIT", "8.0")),
        help="GPU memory limit in GB",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.size <= 0:
        parser.error("Matrix size must be positive")
    if args.iterations <= 0:
        parser.error("Iterations must be positive")
    if args.power < 0.1 or args.power > 2.0:
        parser.error("Power level must be between 0.1 and 2.0")
    if args.gpu_temp_limit <= 0:
        parser.error("GPU temperature limit must be positive")
    if args.memory_limit <= 0:
        parser.error("Memory limit must be positive")

    return args


def main() -> None:
    """Main entry point for the benchmark tool."""
    try:
        # Parse arguments
        args = parse_args()

        # Create configuration
        config = BenchmarkConfig(
            matrix_size=args.size,
            iterations=args.iterations,
            power_level=args.power,
            monitor=args.monitor,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            gpu_temp_limit=args.gpu_temp_limit,
            memory_limit_gb=args.memory_limit,
        )

        # Initialize and run benchmark
        benchmark = CUDABenchmark(config)
        results = benchmark.run_benchmark()

        # Log summary
        logger.info("Benchmark completed successfully")
        logger.info(f"Total time: {results['performance']['total_time']:.2f} seconds")
        logger.info(
            f"Operations per second: {results['performance']['operations_per_second']:.2f}"
        )

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
