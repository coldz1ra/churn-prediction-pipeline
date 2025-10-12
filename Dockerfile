FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install -U pip && pip install -r requirements.txt

COPY . ./

# Expose port
EXPOSE 8000

# Default command: start API if artifacts exist
CMD ["/bin/sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port 8000"]
