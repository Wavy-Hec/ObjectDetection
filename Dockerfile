# FlowCount dashboard image — synthetic demo by default, so it runs with no
# model download, camera, or GPU. Port 7860 makes it Hugging Face Spaces
# (Docker Space) compatible as-is.
#
#   docker build -t flowcount .
#   docker run -p 7860:7860 flowcount
#
# Single worker only: DashboardEngine holds in-process state (counts, heatmap),
# so scaling out workers would give each its own diverging dashboard.

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY flowcount ./flowcount
COPY config.yaml ./

# opencv-python-headless replaces opencv-python: slim images ship no libGL,
# and the dashboard never opens display windows.
RUN pip install --no-cache-dir ".[web]" \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir opencv-python-headless

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/healthz').status == 200 else 1)"

CMD ["uvicorn", "flowcount.web.server:app", "--host", "0.0.0.0", "--port", "7860"]
