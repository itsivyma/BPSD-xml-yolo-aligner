FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BPSD_ALIGNER_JOB_DIR=/var/lib/bpsd-aligner

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .

RUN groupadd --system bpsd \
    && useradd --system --gid bpsd --home-dir /nonexistent bpsd \
    && mkdir -p /var/lib/bpsd-aligner \
    && chown -R bpsd:bpsd /var/lib/bpsd-aligner

USER bpsd
VOLUME ["/var/lib/bpsd-aligner"]

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1
CMD ["bpsd-aligner", "web", "--server.address=0.0.0.0", "--server.port=8501"]
