FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Use 1 worker to reduce memory
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1"]
