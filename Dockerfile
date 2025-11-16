# Multi-stage build for optimized Docker image
FROM python:3.12.7-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --user \
    mlflow==2.19.0 \
    scikit-learn==1.6.0 \
    pandas==2.2.3 \
    numpy==2.2.1 \
    joblib \
    cloudpickle==3.1.0

# Production stage
FROM python:3.12.7-slim

# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local

# Add local binaries to PATH
ENV PATH=/root/.local/bin:$PATH

# Set memory limits to prevent OOM
ENV MALLOC_ARENA_MAX=2
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Set working directory
WORKDIR /opt/ml/model

# Expose port
EXPOSE 8080

# Command to start the MLflow model server
CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "8080", "--env-manager", "local"]
