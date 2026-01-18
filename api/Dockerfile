FROM python:3.12.3
WORKDIR /app
COPY api/ .
RUN pip install fastapi uvicorn tensorflow pillow
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]