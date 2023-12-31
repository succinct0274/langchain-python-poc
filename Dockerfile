FROM python:3.11.4 as build-base

RUN apt update -y && apt upgrade -y

RUN apt install ffmpeg libsm6 libxext6  -y

COPY . /app/chatbot-server

WORKDIR /app/chatbot-server

RUN pip install -r requirements.txt

# RUN python ./script/download_model.py

EXPOSE 8080

CMD uvicorn app.main:app --host 0.0.0.0 --port 8080