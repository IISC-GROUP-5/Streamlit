# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to replicate the local environment
ENV PYTHONPATH=/app

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "gui/Inference_GUI_Streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

