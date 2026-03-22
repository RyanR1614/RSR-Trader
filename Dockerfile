FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLP models
RUN python -m nltk.downloader stopwords wordnet omw-1.4 vader_lexicon -q && \
    python -m spacy download en_core_web_sm -q

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/sentiment models/saved logs plots

# Environment variables (override at runtime)
ENV NEWSAPI_KEY=""
ENV REDDIT_CLIENT_ID=""
ENV REDDIT_SECRET=""
ENV AV_KEY=""

# Default: run the scheduler (for Railway/always-on deployments)
# Override CMD to run daily_run.py for GCP Cloud Run one-shot jobs
CMD ["python", "pipeline/scheduler.py"]
