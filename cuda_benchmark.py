import torch
import time
import argparse


def run_intensive_cuda_operation(matrix_size, iterations, power_level=1.0):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed."
        )

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Iterations: {iterations}")
    print(f"Power level: {power_level}")

    # Create large random matrices on GPU
    a = torch.randn(matrix_size, matrix_size, device="cuda")
    b = torch.randn(matrix_size, matrix_size, device="cuda")

    # Warm up GPU
    torch.matmul(a, b)
    torch.cuda.synchronize()

    start_time = time.time()

    for i in range(iterations):
        # Matrix multiplication with adjustable intensity
        c = torch.matmul(a, b)
        # Additional operations scaled by power_level
        for _ in range(int(power_level * 3)):  # More operations at higher power levels
            c = torch.sin(c)
            c = torch.exp(torch.clamp(c, -1, 1))  # Clamp to prevent overflow

        torch.cuda.synchronize()

        if (i + 1) % max(1, iterations // 10) == 0:
            print(f"Completed iteration {i + 1}/{iterations}")
            current_time = time.time()
            print(f"Time elapsed: {current_time - start_time:.2f} seconds")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

    # Free up GPU memory
    del a, b, c
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Run intensive CUDA operations with customizable parameters"
    )

    parser.add_argument(
        "-s", "--size", type=int, default=10000, help="Matrix size (N for NxN matrix)"
    )

    parser.add_argument(
        "-i", "--iterations", type=int, default=100, help="Number of iterations to run"
    )

    parser.add_argument(
        "-p",
        "--power",
        type=float,
        default=1.0,
        help="Power level (0.1 to 2.0) to adjust computational intensity",
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable detailed GPU monitoring (requires nvidia-smi)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.size <= 0:
        parser.error("Matrix size must be positive")
    if args.iterations <= 0:
        parser.error("Iterations must be positive")
    if args.power < 0.1 or args.power > 2.0:
        parser.error("Power level must be between 0.1 and 2.0")

    if args.monitor:
        try:
            import subprocess
            import threading

            def monitor_gpu():
                while True:
                    subprocess.run(["nvidia-smi"])
                    time.sleep(5)

            monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
            monitor_thread.start()
        except Exception as e:
            print(f"Could not start GPU monitoring: {e}")

    try:
        run_intensive_cuda_operation(
            matrix_size=args.size, iterations=args.iterations, power_level=args.power
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
