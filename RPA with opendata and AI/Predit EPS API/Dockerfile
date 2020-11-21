FROM tensorflow/tensorflow:latest

RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip python3-venv

#for python api
RUN pip3 install --upgrade pip 

# copy file to /app
COPY . /app/

RUN pip3 install numpy
RUN pip3 --no-cache-dir install -r /app/requirements.txt

WORKDIR /app

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV FLASK_APP=api.py
CMD flask run --host=0.0.0.0
