FROM ubuntu:18.04

# Setup the system for installs
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
RUN apt-get update

# Install Java
RUN apt-get install -y openjdk-11-jre

# Install python 3.6
RUN apt-get install -y python3-dev
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Install program requirements
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install spaCy (must be done after numpy) and a language model
RUN pip install -U pip setuptools wheel
RUN pip install spacy==2.3.4
RUN python -m spacy download en_core_web_sm

COPY src .