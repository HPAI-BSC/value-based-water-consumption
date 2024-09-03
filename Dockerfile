FROM ubuntu:mantic-20230807.1

RUN apt-get update && apt-get install -y xmlstarlet python3 python3-pip python-is-python3 openjdk-19-jdk wget graphviz

ARG RESULTS_PATH="/code/output"
ARG MODEL_NAME="PROGRAMA v8.nlogo"
ARG NETLOGO_VERSION=6.3.0
ARG NETLOGO_NAME=NetLogo-$NETLOGO_VERSION
ARG NETLOGO_URL=https://ccl.northwestern.edu/netlogo/$NETLOGO_VERSION/$NETLOGO_NAME-64.tgz
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN mkdir /netlogo \
 && wget $NETLOGO_URL \
 && tar xzf $NETLOGO_NAME-64.tgz -C /netlogo --strip-components=1 \
 && rm $NETLOGO_NAME-64.tgz

COPY requirements.txt requirements.txt
RUN pip3 install --break-system-packages -r requirements.txt
    
COPY script.sh /code/script.sh
COPY setup.xml /code/setup.xml
# COPY setup-sweep.xml /code/setup.xml
# COPY setup_test.xml /code/setup-sweep.xml
COPY $MODEL_NAME /code/NLModel.nlogo
COPY src /code/python-scripts

WORKDIR /code

RUN mkdir -p /gpfs/projects/bsc70/hpai/storage/data/water/results/
ENV RESULTS_PATH=${RESULTS_PATH}
RUN mkdir -p ${RESULTS_PATH}

ENTRYPOINT ["/bin/bash", "/code/script.sh"]
