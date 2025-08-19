"""
App Store Review Analysis Pipeline

This module contains the complete processing pipeline for analyzing Apple App Store reviews.
Separated from the API layer for better modularity and reusability.
"""

import os
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

# Import our modules  
from scraper import fetch_reviews_multi_country, normalize_dataframe, write_outputs, ScraperError
from processor import process_csv_file, ProcessorError  
from translation import translate_csv_file, TranslationError
from sentiment import analyze_sentiment_csv, SentimentError
from composer import ReviewComposer, ComposerError
from insights import ReviewInsightsGenerator, InsightsError


class ReviewAnalysisPipeline:
    """
    Complete pipeline for analyzing Apple App Store reviews.
    
    This class encapsulates all the processing steps from scraping to insights generation.
    """
    
    def __init__(self, storage_dir: Path = None):
        """
        Initialize the pipeline.
        
        Args:
            storage_dir: Directory to store analysis results. Defaults to 'storage'
        """
        self.storage_dir = storage_dir or Path("storage")
        self.storage_dir.mkdir(exist_ok=True)
    
    def run_complete_analysis(
        self,
        app_id: Optional[str] = None,
        app_name: Optional[str] = None,
        countries: Optional[List[str]] = None,
        max_reviews_per_country: int = 100,
        min_topic_size: int = 5,
        openai_api_key: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            app_id: Apple App Store app ID (optional if app_name provided)
            app_name: App name to search for (optional if app_id provided)
            countries: List of country codes (default: ['us'])
            max_reviews_per_country: Maximum reviews per country
            min_topic_size: Minimum reviews per topic for analysis
            openai_api_key: Optional OpenAI API key for insights generation
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing analysis results and file paths
            
        Raises:
            Various error types depending on which stage fails
        """
        countries = countries or ['us']
        
        # Validate inputs
        if not app_id and not app_name:
            raise ValueError("Either app_id or app_name must be provided")
        
        # Create unique analysis directory (use app_id if available, or timestamp if app_name)
        if app_id:
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{app_id}"
        else:
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{app_name.replace(' ', '_').lower()}"
        analysis_dir = self.storage_dir / analysis_id
        analysis_dir.mkdir(exist_ok=True)
        
        try:
            # Stage 1: Scraping
            if progress_callback:
                progress_callback("Collecting reviews from App Store...")
            print(f"📱 Collecting reviews from {countries}...")
            
            # Convert app_id to int if provided, otherwise pass None
            app_id_int = int(app_id) if app_id else None
            
            raw_reviews = fetch_reviews_multi_country(
                app_id_int,
                app_name,
                countries,
                max_reviews_per_country
            )
            
            if not raw_reviews:
                raise ScraperError("No reviews found for this app")
            
            # Normalize and save raw data
            df = normalize_dataframe(raw_reviews)
            
            # Extract app info from the data
            app_info = {
                "app_id": app_id or (df['app_id'].iloc[0] if 'app_id' in df.columns else None),
                "app_name": app_name or (df['app_name'].iloc[0] if 'app_name' in df.columns and not df['app_name'].isna().iloc[0] else None),
                "countries": countries,
                "total_reviews": len(df),
                "scraped_at": datetime.now(timezone.utc).isoformat()
            }
            
            raw_file_path, meta_file_path = write_outputs(
                df,
                analysis_dir,
                "raw_reviews",
                app_info
            )
            
            print(f"✅ Collected {len(df)} reviews")
            if progress_callback:
                progress_callback(f"Collected {len(df)} reviews")
            
            # Stage 2: Text Processing
            if progress_callback:
                progress_callback("Processing text...")
            print("🔤 Processing text...")
            
            processed_file = analysis_dir / "processed_reviews.csv"
            process_csv_file(raw_file_path, processed_file)
            
            # Stage 3: Translation
            if progress_callback:
                progress_callback("Translating reviews...")
            print("🌐 Translating reviews...")
            
            translated_file = analysis_dir / "translated_reviews.csv"
            try:
                translate_csv_file(processed_file, translated_file)
                print("✅ Translation completed")
            except Exception as e:
                print(f"⚠️ Warning: Translation failed: {e}")
                print(f"📋 Debug: Will copy processed file as translated file for analysis continuation")
                # Copy processed file as fallback
                import shutil
                shutil.copy2(processed_file, translated_file)
                # Add basic language columns
                df_fallback = pd.read_csv(translated_file)
                df_fallback['detected_language'] = 'en'  # Assume English
                df_fallback['language_confidence'] = 1.0
                df_fallback['review_clean_translated'] = df_fallback['review_clean']  # Use original
                df_fallback.to_csv(translated_file, index=False)
                print("✅ Created fallback translated file")
            
            # Stage 4: Sentiment Analysis
            if progress_callback:
                progress_callback("Analyzing sentiment...")
            print("😊 Analyzing sentiment...")
            
            sentiment_file = analysis_dir / "sentiment_reviews.csv"
            analyze_sentiment_csv(translated_file, sentiment_file)
            
            # Stage 5: Composition and Topic Modeling
            if progress_callback:
                progress_callback("Discovering topics...")
            print("🎯 Discovering topics...")
            
            composer = ReviewComposer(min_topic_size=min_topic_size)
            df_final = pd.read_csv(sentiment_file)
            composed_data = composer.compose_analysis(df_final)
            
            composed_file = analysis_dir / "composed_analysis.json"
            with open(composed_file, 'w', encoding='utf-8') as f:
                json.dump(composed_data, f, ensure_ascii=False, indent=2)
            
            # Stage 6: AI Insights (optional)
            insights_data = None
            insights_file_path = None
            
            if openai_api_key:
                try:
                    if progress_callback:
                        progress_callback("Generating AI insights...")
                    print("🧠 Generating AI insights...")
                    print(f"📋 Debug: Using OpenAI API key: {openai_api_key[:12]}...{openai_api_key[-8:] if len(openai_api_key) > 20 else '***'}")
                    
                    insights_generator = ReviewInsightsGenerator(api_key=openai_api_key)
                    insights_data = insights_generator.generate_comprehensive_insights(composed_data)
                    
                    insights_file = analysis_dir / "insights.json"
                    with open(insights_file, 'w', encoding='utf-8') as f:
                        json.dump(insights_data, f, ensure_ascii=False, indent=2)
                    insights_file_path = str(insights_file)
                    print("✅ AI insights generated successfully")
                    
                except Exception as e:
                    print(f"⚠️ Warning: AI insights generation failed: {e}")
                    print(f"📋 Debug: Error type: {type(e).__name__}")
                    if hasattr(e, 'response'):
                        print(f"📋 Debug: API response: {e.response}")
                    if progress_callback:
                        progress_callback(f"Warning: AI insights generation failed: {e}")
            else:
                print("ℹ️ No OpenAI API key provided - skipping AI insights generation")
            
            # Extract key metrics from composed data
            composition_info = composed_data["composition_info"]
            insights_prep = composed_data["insights_preparation"]
            summary = composed_data["summary"]
            
            # Build response
            result = {
                "analysis_id": analysis_id,
                "analysis_dir": str(analysis_dir),
                "total_reviews": composition_info["total_reviews"],
                "negative_reviews": composition_info["negative_reviews"],
                "topics_discovered": composition_info["topics_discovered"],
                "metrics": {
                    "rating_distribution": insights_prep["overview"]["review_distribution"],
                    "sentiment_distribution": insights_prep["overview"]["sentiment_breakdown"],
                    "country_breakdown": insights_prep["overview"]["app_info"].get("countries", []),
                    "main_complaints": summary["key_findings"]["biggest_complaint_themes"][:3] if summary["key_findings"]["biggest_complaint_themes"] else [],
                    "negative_percentage": round(composition_info["negative_reviews"] / composition_info["total_reviews"] * 100, 1) if composition_info["total_reviews"] > 0 else 0
                },
                "insights": insights_data,
                "files": {
                    "raw_reviews": str(raw_file_path),
                    "raw_meta": str(meta_file_path),
                    "processed_reviews": str(processed_file),
                    "translated_reviews": str(translated_file),
                    "sentiment_reviews": str(sentiment_file),
                    "composed_analysis": str(composed_file),
                    "insights": insights_file_path
                },
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Save summary for later retrieval
            summary_file = analysis_dir / "summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            if progress_callback:
                progress_callback("Analysis completed successfully!")
            print(f"🎉 Analysis completed! Results saved to {analysis_dir}")
            
            return result
            
        except Exception as e:
            # Clean up on failure
            import shutil
            if analysis_dir.exists():
                shutil.rmtree(analysis_dir)
            raise e
    
    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a previous analysis by ID.
        
        Args:
            analysis_id: The analysis ID to retrieve
            
        Returns:
            Analysis data if found, None otherwise
        """
        analysis_dir = self.storage_dir / analysis_id
        summary_file = analysis_dir / "summary.json"
        
        if not summary_file.exists():
            return None
            
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def list_analyses(self) -> List[Dict[str, Any]]:
        """
        List all available analyses.
        
        Returns:
            List of analysis summaries
        """
        analyses = []
        
        for analysis_dir in self.storage_dir.iterdir():
            if analysis_dir.is_dir() and analysis_dir.name.startswith('analysis_'):
                summary_file = analysis_dir / "summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        
                        # Extract app_name from various sources
                        app_name = None
                        
                        # Try to get from metadata first
                        raw_meta_file = analysis_dir / f"raw_reviews_{analysis_dir.name.split('_')[1]}-{analysis_dir.name.split('_')[2][:6]}.meta.json"
                        if raw_meta_file.exists():
                            try:
                                with open(raw_meta_file, 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                                app_name = meta.get("app_name")
                            except:
                                pass
                        
                        # If not in metadata, try to extract app_id and get from title if it contains app name
                        if not app_name:
                            app_id = None
                            if "app_id" in summary:
                                app_id = summary["app_id"]
                            elif analysis_dir.name.split('_')[-1].isdigit():
                                app_id = analysis_dir.name.split('_')[-1]
                            
                            # For known popular apps, provide names
                            app_id_to_name = {
                                "835599320": "TikTok",
                                "389801252": "Instagram", 
                                "310633997": "WhatsApp",
                                "363590051": "Netflix",
                                "324684580": "Spotify"
                            }
                            app_name = app_id_to_name.get(app_id, f"App {app_id}" if app_id else "Unknown App")
                        
                        analyses.append({
                            "analysis_id": analysis_dir.name,
                            "app_name": app_name,
                            "total_reviews": summary.get("total_reviews", 0),
                            "negative_reviews": summary.get("negative_reviews", 0),
                            "topics_discovered": summary.get("topics_discovered", 0),
                            "processed_at": summary.get("processed_at"),
                            "has_insights": summary.get("insights") is not None,
                            "countries": summary.get("metrics", {}).get("country_breakdown", [])
                        })
                    except:
                        continue
        
        # Sort by processed_at descending
        analyses.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
        return analyses
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete an analysis and its files.
        
        Args:
            analysis_id: The analysis ID to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        analysis_dir = self.storage_dir / analysis_id
        
        if not analysis_dir.exists() or not analysis_dir.is_dir():
            return False
        
        try:
            import shutil
            shutil.rmtree(analysis_dir)
            return True
        except Exception:
            return False
    
    def get_file_path(self, analysis_id: str, file_type: str) -> Optional[Path]:
        """
        Get the path to a specific file from an analysis.
        
        Args:
            analysis_id: The analysis ID
            file_type: Type of file ('raw_reviews', 'processed_reviews', etc.)
            
        Returns:
            Path to the file if it exists, None otherwise
        """
        analysis = self.get_analysis_by_id(analysis_id)
        if not analysis or "files" not in analysis:
            return None
        
        file_path_str = analysis["files"].get(file_type)
        if not file_path_str:
            return None
            
        file_path = Path(file_path_str)
        return file_path if file_path.exists() else None


# Convenience function for backward compatibility
def run_analysis_pipeline(
    app_id: str,
    countries: Optional[List[str]] = None,
    max_reviews_per_country: int = 100,
    min_topic_size: int = 5,
    openai_api_key: Optional[str] = None,
    storage_dir: Path = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the complete analysis pipeline.
    
    This is a simple wrapper around ReviewAnalysisPipeline.run_complete_analysis()
    for backward compatibility and ease of use.
    """
    pipeline = ReviewAnalysisPipeline(storage_dir)
    return pipeline.run_complete_analysis(
        app_id=app_id,
        countries=countries,
        max_reviews_per_country=max_reviews_per_country,
        min_topic_size=min_topic_size,
        openai_api_key=openai_api_key,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    # Example usage
    pipeline = ReviewAnalysisPipeline()
    
    # Example: Analyze TikTok reviews
    result = pipeline.run_complete_analysis(
        app_id="835599320",  # TikTok
        countries=["us"],
        max_reviews_per_country=50,
        min_topic_size=5
    )
    
    print(f"Analysis completed: {result['analysis_id']}")
    print(f"Total reviews: {result['total_reviews']}")
    print(f"Negative percentage: {result['metrics']['negative_percentage']}%")