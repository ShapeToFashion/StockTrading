# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV FLASK_ENV=development

COPY . .

CMD [ "python" , "-m" , "flask" , "run" , "--host=0.0.0.0"]

