# Food101 Image Classification – End-to-End ML App

This project demonstrates an end-to-end machine learning workflow using TensorFlow, FastAPI, and Streamlit.

## Overview
- Progressive transfer learning on Food101 (TFDS)
- Model exported as TensorFlow SavedModel
- Production-accurate evaluation with cached predictions
- Error analysis (per-class, confusion pairs, misclassification gallery)
- FastAPI inference service
- Streamlit interactive dashboard

## Project Structure
- `notebooks/` – training, evaluation, and analysis
- `artifacts/` – SavedModel and cached evaluation results
- `api/` – FastAPI inference service
- `streamlit_app/` – Streamlit UI

## Run Locally

### FastAPI
```bash
uvicorn api.main:app --reload

### Streamlit
streamlit run streamlit_app/streamlit_app.py