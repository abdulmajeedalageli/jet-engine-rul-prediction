FROM python:3.13-slim

WORKDIR /app

# Dependencies first: this layer is reused on rebuilds unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs: no data, notebook, tests or training code paths are used at serve time.
COPY src/ src/
COPY models/ models/
COPY api.py .

# Don't run the server as root inside the container.
RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
