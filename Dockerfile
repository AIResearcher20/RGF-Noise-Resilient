FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH=/app

CMD ["python", "run.py", "--config", "configs/default.yaml"]
