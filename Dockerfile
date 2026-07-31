FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[catboost,app]"
EXPOSE 8501
CMD ["streamlit", "run", "src/semiyield/streamlit_app.py", "--server.address=0.0.0.0"]

