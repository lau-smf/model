FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.23.2 \
    pydantic==2.4.2 \
    python-dotenv==1.0.0 \
    langchain==0.0.312 \
    llama-cpp-python==0.2.11

# Create directories
RUN mkdir -p /app/models

# Download the Zephyr model from Hugging Face
RUN wget https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_0.gguf \
    -O /app/models/zephyr-7b-beta.Q5_0.gguf

# Copy application code
COPY main.py /app/

# Set environment variables
ENV MODEL_PATH=/app/models/zephyr-7b-beta.Q5_0.gguf

# Expose port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]