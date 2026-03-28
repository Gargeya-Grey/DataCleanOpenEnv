FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai pytest requests openenv-core

# Copy source code
COPY . .

# Install the project itself in editable mode or just make sure PYTHONPATH is set
RUN pip install -e .

EXPOSE 8000

# Run the server entry point defined in pyproject.toml
CMD ["python", "-m", "server.app"]
