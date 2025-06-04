# Use a lightweight Python base image
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables to avoid "Streamlit cannot run as root" issue
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false

# Command to run the app
CMD ["streamlit", "run", "home.py"]
