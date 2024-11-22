# Streamlit Application with Docker Deployment

This project is a Dockerized Streamlit application that supports batch and manual predictions with the ability to upload custom machine learning models.

---

## Features

- **Streamlit GUI**: User-friendly interface for manual and batch predictions.
- **Dynamic Model Loading**: Upload new models via the app.
- **Dockerized Deployment**: Seamless deployment using Docker.
- **Custom Python Path**: Configured to use a custom `PYTHONPATH` for module imports.

---

## Prerequisites

1. **Docker** (Recommended for deployment).
2. **Python 3.9+** (For local setup).

---
## Getting Started

### Option 1: Local Setup (Without Docker)

1. **Clone the Repository**:
   ```bash
   git clone [<repository-url>](https://github.com/IISC-GROUP-5/Streamlit.git)
   cd Streamlit

2. **Virtual Environment**
   ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    pip install -r requirements.txt

3. ```bash
    PYTHONPATH=$(pwd) streamlit run gui/Inference_GUI_Streamlit.py

###  Option 2: Dockerized Setup

1. **Build the Docker Image**:
   ```bash
    docker build -t streamlit-app .

2. **Run the Docker Container**:
    ```bash
    docker run -p 8501:8501 streamlit-app

3. **Access the Application**:
Open http://localhost:8501 in your web browser.


