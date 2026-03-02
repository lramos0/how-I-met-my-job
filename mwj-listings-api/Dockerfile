FROM python:3.11-slim

WORKDIR /opt/app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

EXPOSE 6001

# Listen on 0.0.0.0 so ALB/host can reach it
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6001"]

