FROM python:3.11.4 as build-base

RUN apt update -y && apt upgrade -y

COPY . /app/chatbot-server

WORKDIR /app/chatbot-server

RUN pip install -r requirements.txt

RUN python ./script/download_model.py

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0