FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8"

RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
      python3.6 jq curl libglib2.0-0 libsm6 libxrender1 libxext6 && \
    ln -s /usr/bin/python3.6 /usr/bin/python && \
    curl -L "https://bootstrap.pypa.io/get-pip.py" | python3.6

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

COPY ./docker/entrypoint.sh ./
COPY ./ssd ./ssd

ENTRYPOINT ["./entrypoint.sh"]
