# Use the official Python slim image as the base
FROM python:3.10.15-slim

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    g++ \
    git \
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools, then install Poetry
RUN pip install --upgrade pip setuptools wheel && \
    pip install poetry

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first (to leverage Docker's layer caching)
COPY poetry.lock pyproject.toml .

# Install Python dependencies using Poetry
RUN poetry run pip install cython ipykernel && \
    poetry run pip install --no-use-pep517 youtokentome

# Install Hugging Face CLI, < 0.24 will throw ModelFilter error when loading reazonspeech nemo model
RUN poetry run pip install -U huggingface_hub[cli]==0.23.2

# Configure Git to store credentials globally
RUN git config --global credential.helper store

# Install ReazonSpeech models, Clone the repo first
RUN git clone https://github.com/reazon-research/ReazonSpeech
# NeMo Model
RUN poetry run pip install ReazonSpeech/pkg/nemo-asr
# K2 Model
RUN poetry run pip install ReazonSpeech/pkg/k2-asr
# ESPnet model
RUN poetry run pip install ReazonSpeech/pkg/espnet-asr
# Add a placeholder for Hugging Face token login (replace with your token)
# RUN huggingface-cli login --token $HF_TOKEN

# Set default command to bash for interactive usage (optional)
CMD ["bash"]