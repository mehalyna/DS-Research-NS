# Start with a lightweight Python 3.10 installation
FROM python:3.10-slim

# Set the folder inside the container where we will work
WORKDIR /app

# Install system tools needed for Postgres and some ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from your computer to the container
COPY requirements.txt .

# Install the Python libraries (Django, DRF, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend code
COPY . .