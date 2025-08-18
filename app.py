import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Import our pipeline
from pipeline import ReviewAnalysisPipeline, ScraperError, ProcessorError, TranslationError, SentimentError, ComposerError, InsightsError
from scraper import search_apps_by_name


# Pydantic models for API requests/responses
class ReviewCollectionRequest(BaseModel):
    app_id: str = Field(..., description="Apple App Store app ID")
    countries: Optional[List[str]] = Field(default=None, description="List of country codes (default: ['us'])")
    max_reviews_per_country: int = Field(default=100, description="Maximum reviews per country")
    min_topic_size: int = Field(default=5, description="Minimum reviews per topic for analysis")

class AnalysisResponse(BaseModel):
    total_reviews: int
    negative_reviews: int
    topics_discovered: int
    metrics: Dict[str, Any]
    insights: Optional[Dict[str, Any]] = None
    files: Dict[str, Optional[str]]
    processed_at: str


# FastAPI app
app = FastAPI(
    title="App Store Review Insights API",
    description="Simple synchronous API for collecting and analyzing Apple App Store reviews",
    version="1.0.0"
)

# Storage directory
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "App Store Review Insights API (Simple)",
        "version": "1.0.0",
        "description": "Synchronous analysis of Apple App Store reviews",
        "endpoints": {
            "analyze": "POST /analyze - Complete analysis pipeline",
            "health": "GET /health - Health check"
        }
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_reviews(
    request: ReviewCollectionRequest,
    openai_api_key: Optional[str] = Query(None, description="OpenAI API key for insights generation")
):
    """
    Perform complete review analysis using the pipeline.
    This processes everything in sequence and returns final results.
    """
    try:
        print(f"Starting analysis for app {request.app_id}")
        
        # Initialize pipeline
        pipeline = ReviewAnalysisPipeline(STORAGE_DIR)
        
        # Run analysis using pipeline
        result = pipeline.run_complete_analysis(
            app_id=request.app_id,
            countries=request.countries,
            max_reviews_per_country=request.max_reviews_per_country,
            min_topic_size=request.min_topic_size,
            openai_api_key=openai_api_key
        )
        
        # Convert result to API response format
        response_data = {
            "total_reviews": result["total_reviews"],
            "negative_reviews": result["negative_reviews"],
            "topics_discovered": result["topics_discovered"],
            "metrics": result["metrics"],
            "insights": result["insights"],
            "files": result["files"],
            "processed_at": result["processed_at"]
        }
        
        return AnalysisResponse(**response_data)
        
    except ScraperError as e:
        raise HTTPException(status_code=400, detail=f"Scraping error: {str(e)}")
    except ProcessorError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except TranslationError as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")
    except SentimentError as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")
    except ComposerError as e:
        raise HTTPException(status_code=500, detail=f"Analysis composition error: {str(e)}")
    except InsightsError as e:
        raise HTTPException(status_code=500, detail=f"Insights generation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/download/{analysis_id}/{file_type}")
async def download_file(analysis_id: str, file_type: str):
    """Download analysis files by analysis ID and file type."""
    pipeline = ReviewAnalysisPipeline(STORAGE_DIR)
    
    # Get file path from pipeline
    file_path = pipeline.get_file_path(analysis_id, file_type)
    
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check - ensure file is in storage directory
    try:
        file_path.resolve().relative_to(STORAGE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Determine media type and filename
    if file_path.suffix == '.csv':
        media_type = 'text/csv'
    elif file_path.suffix == '.json':
        media_type = 'application/json'
    else:
        media_type = 'application/octet-stream'
    
    # Create a descriptive filename
    app_id_match = analysis_id.split('_')[-1] if '_' in analysis_id else 'unknown'
    if file_type == 'raw_reviews':
        filename = f"raw_reviews_{app_id_match}.csv"
    elif file_type == 'raw_meta':
        filename = f"raw_reviews_metadata_{app_id_match}.json"
    else:
        filename = f"{file_type}_{app_id_match}{file_path.suffix}"
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )

@app.get("/download/{file_path:path}")
async def download_file_legacy(file_path: str):
    """Legacy download endpoint - downloads by full file path."""
    full_path = Path(file_path)
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check - ensure file is in storage directory
    try:
        full_path.resolve().relative_to(STORAGE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Determine media type
    if full_path.suffix == '.csv':
        media_type = 'text/csv'
    elif full_path.suffix == '.json':
        media_type = 'application/json'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        path=str(full_path),
        media_type=media_type,
        filename=full_path.name
    )

@app.get("/analyses")
async def list_analyses():
    """List all analysis directories."""
    pipeline = ReviewAnalysisPipeline(STORAGE_DIR)
    analyses = pipeline.list_analyses()
    return {"analyses": analyses}

@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis and its files."""
    pipeline = ReviewAnalysisPipeline(STORAGE_DIR)
    
    if not pipeline.delete_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {"message": f"Analysis {analysis_id} deleted successfully"}

@app.get("/search")
async def search_apps(
    query: str = Query(..., description="App name to search for"),
    country: str = Query("us", description="Country code (e.g., 'us', 'gb')"),
    limit: int = Query(10, description="Maximum number of results", ge=1, le=50)
):
    """Search for apps by name."""
    try:
        results = search_apps_by_name(query, country, limit)
        return {"query": query, "country": country, "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "storage_dir": str(STORAGE_DIR),
        "total_analyses": len([d for d in STORAGE_DIR.iterdir() if d.is_dir() and d.name.startswith('analysis_')])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)