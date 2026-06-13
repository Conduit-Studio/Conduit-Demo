# BYO SageMaker training image for YOLO (ultralytics) fine-tuning.
#
# This is the ONE container Conduit does NOT build for you: the Train Model node
# points at this image by digest. Run Code nodes (e.g. select_best) get built from
# their requirements.txt by Conduit — only Train Model needs a hand-built image.
#
# Build + push, then PIN the @sha256 digest in the Train Model node:
#     bash scripts/build_and_push.sh
#
# SageMaker BYO-training contract (legacy mode): SageMaker runs `docker run <image> train`.
# Our ENTRYPOINT runs train.py and ignores that extra "train" argv.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # ultralytics: keep it headless + writable + no phone-home.
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLBACKEND=Agg

# System libraries OpenCV (pulled in by ultralytics) needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

COPY train.py /opt/ml/code/train.py
WORKDIR /opt/ml/code

ENTRYPOINT ["python", "train.py"]
