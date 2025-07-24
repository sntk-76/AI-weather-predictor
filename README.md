![Cover](https://github.com/sntk-76/AI-weather-predictor/blob/main/project_plan/other/cover.png?raw=true)

# AI Weather Forecasting System

## Overview

This project is a production-grade, end-to-end machine learning pipeline that forecasts 7-day weather conditions for cities worldwide. It integrates modern MLOps tools, cloud infrastructure, deep learning architectures, a scalable preprocessing pipeline, and a conversational LLM interface. The final product is exposed via a web application that accepts user input and returns a human-readable forecast summary.

**Access the live application here:**  
👉 [https://ai-weather-predictor.streamlit.app/](https://ai-weather-predictor.streamlit.app/)


---

## Project Objective

The objective of this project is to demonstrate the full machine learning lifecycle — from data collection and model development to cloud deployment and user interface — within a real-world forecasting problem. The system is designed for scalability, reproducibility, and ease of use, fulfilling all course and industry-standard best practices.

---

## Problem Statement

Accurate weather forecasting is critical for planning in agriculture, logistics, travel, and disaster management. Traditional weather models often lack city-level personalization and interpretability for the general public. This project aims to fill that gap by:

* Aggregating historical weather data from multiple global stations
* Building a deep learning model capable of multi-day, multi-feature forecasting
* Enabling real-time, on-demand predictions for any global city
* Providing human-readable summaries using large language models (LLMs)

---

## Key Features

| Component               | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| **Data Source**         | Historical weather data (2010–2024) via Meteostat API                     |
| **Preprocessing**       | Time enrichment, missing value imputation, engineered weather metrics     |
| **Model**               | Conv1D + GRU deep learning sequence-to-sequence model                     |
| **Experiment Tracking** | MLflow with parameter logging, metric tracking, and artifact registry     |
| **Deployment**          | Streamlit web application with real-time inference and LLM integration    |
| **Monitoring**          | Google Sheets logging of search history, evaluation metrics visualization |
| **Cloud Integration**   | Terraform-provisioned GCP infrastructure (GCS + BigQuery)                 |

---

## Project Architecture

```
User Input
    ↓
Streamlit Web App
    ↓
City Geolocation + Meteostat Data Fetch
    ↓
Preprocessing (feature + sequence generation)
    ↓
Model Inference (Keras)
    ↓
GPT-4o-mini Summary Generation
    ↓
Forecast Table + Text Output
    ↓
Google Sheets Logging
```

---

## Technologies Used

| Category               | Tools / Libraries                      |
| ---------------------- | -------------------------------------- |
| ML/DL                  | TensorFlow, Keras, NumPy, Scikit-learn |
| Data Collection        | Meteostat, Geopy, OpenCage API         |
| Feature Engineering    | Custom Python scripts                  |
| Experiment Tracking    | MLflow                                 |
| Web Interface          | Streamlit                              |
| LLM Summarization      | OpenAI GPT-4o-mini                     |
| Infrastructure as Code | Terraform                              |
| Cloud Platform         | Google Cloud Platform (GCS, BigQuery)  |
| Logging                | Google Sheets API (`gspread`)          |
| Deployment Tools       | Streamlit Cloud, Docker (future scope) |

---

## Setup Instructions

### 1. Environment Setup

```bash
conda create -n weather-forecast python=3.10
conda activate weather-forecast
pip install -r requirements.txt
```

### 2. Download Raw Data

Use the built-in script to download historical weather data:

```python
from data_collection import data_downloader
data = data_downloader()
data.to_csv("data/raw_data.csv", index=False)
```

### 3. Preprocess Data

```python
from preprocessing import main
import pandas as pd

raw_data = pd.read_csv("data/raw_data.csv")
X, y = main(raw_data)
```

### 4. Train Model

```python
from model_training import train_and_save_model
train_and_save_model()
```

### 5. Evaluate Model

```python
from evaluation import evaluate_model
evaluate_model()
```

### 6. Deploy App

```bash
streamlit run app.py
```

---

## Reproducibility

* All model files, training scripts, and preprocessing steps are versioned
* Dependencies are documented in `requirements.txt`
* MLflow logs all metrics and model artifacts
* Terraform scripts provision identical GCP environments
* Data saved in GCS for persistent access

---

## Cloud Infrastructure

Using Terraform, the following components are provisioned on Google Cloud:

* A **Cloud Storage Bucket** with folders for:

  * `raw_data/`
  * `features/`
  * `target/`
* A **BigQuery Dataset** for future analytics workflows
* Configurable via `variables.tf` and securely authenticated using a service account JSON key

---

## Model Performance

* Input shape: `[30, 21]` → Output shape: `[7, 6]`
* Forecast features: `tavg`, `tmin`, `tmax`, `wspd`, `prcp`, `snow`
* Metrics:

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)
* Visual comparison of ground truth vs prediction available in plots

---

## Experiment Tracking

* Managed using MLflow (`/kaggle/working/mlflow`)
* Parameters: epochs, batch size, input/output shapes
* Metrics: `loss`, `val_loss`, `mae`, `val_mae`
* Model artifacts saved and zipped for portability

---

## Web Application Highlights

* Users input a city name
* System fetches real-time weather data and makes 7-day predictions
* GPT-4o-mini provides an intuitive weekly forecast summary
* City searches are logged via Google Sheets for tracking and usage analysis
* Access the live application here:
👉 [https://ai-weather-predictor.streamlit.app/](https://ai-weather-predictor.streamlit.app/)


---

## CI/CD & Best Practices

* Terraform for infrastructure-as-code
* MLflow for experiment reproducibility
* Modular code design (separate scripts for each pipeline stage)
* Logging of model performance and user queries
* Compatible with Docker and CI/CD tools (future extension)

---

## Future Work

* Integrate CI/CD with GitHub Actions
* Enable Docker containerization and deployment to GCP App Engine
* Add automated retraining if model drift detected
* Expand monitoring with tools like Evidently or WhyLogs

---

## License

This project is licensed under the MIT License.

---

Let me know if you'd like this README exported as a PDF, included in your repository as a file, or styled with Markdown badges and sections.
