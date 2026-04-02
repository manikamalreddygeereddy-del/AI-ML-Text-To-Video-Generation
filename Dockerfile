# Use a lightweight Python image
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY Scenes-Generation.py .
COPY Image-Generation.py .

# Run the script when the container starts
CMD ["python3.14", "Scenes-Generation.py"]