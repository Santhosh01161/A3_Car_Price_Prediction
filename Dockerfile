# Use slim Python image
FROM python:3.12

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .
COPY app.py .


# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

# Copy ALL local files (including app.py) into /app
COPY . .

# Expose port
EXPOSE 5000 80

# Run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]