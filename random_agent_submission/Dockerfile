FROM python:3.9-slim-buster

WORKDIR /service
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /service
CMD ["python", "agent.py"]

