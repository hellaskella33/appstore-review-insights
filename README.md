#  App Store Review Insights

A comprehensive tool for analyzing Apple App Store reviews using AI 

##  Features

- ** Complete Review Analysis**: Scrape, process, translate, and analyze App Store reviews
- ** Topic Modeling**: Discover common themes in negative reviews using BERTopic
- ** Sentiment Analysis**: Analyze sentiment using VADER sentiment analysis
- ** Multi-country Support**: Analyze reviews from multiple countries
- ** AI-Powered Insights**: Generate detailed insights using OpenAI GPT
- ** Data Export**: Download raw and processed data in CSV/JSON formats
- ** Web Interface**: User-friendly Streamlit interface
- ** API Access**: RESTful API for programmatic access
- ** Docker Support**: Easy deployment with Docker and docker-compose

##  Quick Start

### Before start we need to create .env file with OPEN_AI_API_KEY

### Option 1: Using Docker (Recommended)

```bash
# Run both API and Streamlit
docker compose up --build
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python app.py

# In another terminal, start Streamlit
streamlit run streamlit_app.py
```

## 📖 Usage

### Using the Web Interface (Streamlit)

1. Open http://localhost:8501 in your browser
2. Enter an App Store ID or App Name (e.g., `835599320` for TikTok)
3. Configure analysis parameters
4. Click "Start Analysis" and wait for results

### Using the API

```bash
# Start analysis
curl -X POST "http://localhost:8000/analyze" \
-H "Content-Type: application/json" \
-d '{
  "app_id": "835599320",
  "countries": ["us"],
  "max_reviews_per_country": 200,
  "min_topic_size": 5
}'

# Download raw reviews
curl "http://localhost:8000/download/{analysis_id}/raw_reviews" -O
```
## 🔍 Popular App Store IDs

- **TikTok**: `835599320`
- **Instagram**: `389801252`  
- **WhatsApp**: `310633997`
- **Netflix**: `363590051`
- **Spotify**: `324684580`

##  Docker Deployment

```bash
# Basic deployment
docker-compose up --build

# API only
docker-compose --profile api-only up api

# Streamlit only  
docker-compose --profile streamlit-only up streamlit
```

**Built for analyzing app store reviews and understanding user feedback**
