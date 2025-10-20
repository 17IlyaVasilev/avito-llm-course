FROM registry.k.avito.ru/nvcr-proxy/nvidia/pytorch:24.12-py3 AS base

WORKDIR /app
RUN python3 -m pip install --upgrade pip \
    && pip install 'transformers==4.52.0' 'datasets==3.6.0' 'wandb==0.19.10' 'accelerate==1.6.0'

RUN python3 -m pip install --no-build-isolation \
    'flash-attn==2.7.3'

ENV HF_DISABLE_TELEMETRY=true \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    SERVICE_HOST=0.0.0.0 \
    AVITO_HTTP_PROXY="http://prx-squid-rev.msk.avito.ru:9090" \
    AVITO_HTTPS_PROXY="http://prx-squid-rev.msk.avito.ru:9090" \
    SERVICE_PORT=8890 \
    PROJECT_ROOT=/app

ENV PATH="$PROJECT_ROOT/bin:$PATH" PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"



COPY . $PROJECT_ROOT