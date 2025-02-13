# Start with NVIDIA's CUDA Alpine base image
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ENV CUDA_BENCHMARK_OUTPUT_DIR=/app/results
ENV CUDA_BENCHMARK_LOG_INTERVAL=10
ENV CUDA_BENCHMARK_TEMP_LIMIT=85
ENV CUDA_BENCHMARK_MEMORY_LIMIT=8.0

# Create output directory
RUN mkdir -p /app/results

# Set volume for persistent results
VOLUME /app/results

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


# Create and set working directory
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy our Python script
COPY cuda_benchmark.py .

# Set default command
ENTRYPOINT ["python3", "cuda_benchmark.py"]

# Default arguments that can be overridden
CMD ["--size", "5000", "--iterations", "50", "--power", "1.0"]