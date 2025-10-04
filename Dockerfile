# Use slim Python image
FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


# Set Python path
ENV PYTHONPATH=/app
# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .
COPY app.py .


# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt
RUN pip install 'dash[testing]' pytest pytest-depends
RUN pip install gunicorn  # explicitly add gunicorn

# Copy ALL local files (including app.py) into /app
COPY . .

# Expose port
EXPOSE 5000 80

# Run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]