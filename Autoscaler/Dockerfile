FROM python:3.10-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY autoscaler.py .

# run app
CMD ["python", "-u", "autoscaler.py"]