# Start with NVIDIA's CUDA Alpine base image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# Create and set working directory
WORKDIR /app

# Copy our Python script
COPY cuda_benchmark.py .

# Set default command
ENTRYPOINT ["python3", "cuda_benchmark.py"]

# Default arguments that can be overridden
CMD ["--size", "5000", "--iterations", "50", "--power", "1.0"]