FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl ffmpeg build-essential gcc g++ libffi-dev libssl-dev libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3", "main.py"]