FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .

EXPOSE 8501
CMD ["bpsd-aligner", "web", "--server.address=0.0.0.0", "--server.port=8501"]
