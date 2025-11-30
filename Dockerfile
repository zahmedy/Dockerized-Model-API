FROM python:3.12-slim

WORKDIR /app

# install dependencies first (cache friendly)
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the est of the app ( model + app)
COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
