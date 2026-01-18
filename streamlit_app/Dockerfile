FROM python:3.12.3
WORKDIR /app
COPY streamlit_app/ .
RUN pip install streamlit requests pandas numpy
CMD ["streamlit", "run", "app.py"]
