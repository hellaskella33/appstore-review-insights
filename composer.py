import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import argparse

import pandas as pd

from topic_modeler import ReviewTopicModeler, TopicModelerError


class ComposerError(Exception):
    pass


class ReviewComposer:
    """
    Data orchestration for review analysis pipeline.
    Coordinates topic modeling and prepares data for insights generation.
    """
    
    def __init__(self, min_topic_size: int = 5, language_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the composer.
        
        Args:
            min_topic_size: Minimum number of reviews to form a topic
            language_model: Sentence transformer model for embeddings
        """
        self.min_topic_size = min_topic_size
        self.language_model = language_model
        self.topic_modeler = ReviewTopicModeler(min_topic_size, language_model)
        
    def compose_analysis(
        self, 
        df: pd.DataFrame,
        sentiment_col: str = "review_sentiment_class",
        text_col: str = "review_clean_translated"
    ) -> Dict[str, Any]:
        """
        Compose complete analysis by running topic modeling and preparing insights data.
        
        Args:
            df: DataFrame with review data including sentiment analysis
            sentiment_col: Column containing sentiment classification  
            text_col: Column containing review text to analyze
            
        Returns:
            Dictionary with composed analysis results
        """
        print("🎼 Starting composed review analysis...")
        
        # Run topic modeling
        print("📊 Performing topic modeling...")
        topic_results = self.topic_modeler.model_topics(df, sentiment_col, text_col)
        
        # Prepare data for insights generation
        print("📝 Preparing data for insights generation...")
        insights_data = self._prepare_insights_data(df, topic_results, sentiment_col, text_col)
        
        # Compose final results
        composed_results = {
            "composition_info": {
                "composed_at": datetime.now(timezone.utc).isoformat(),
                "total_reviews": len(df),
                "negative_reviews": topic_results["modeling_info"]["negative_reviews"],
                "topics_discovered": topic_results["modeling_info"]["topics_discovered"],
                "outliers": topic_results["modeling_info"]["outliers"],
                "text_column_used": topic_results["modeling_info"]["text_column_used"],
                "sentiment_column_used": sentiment_col,
                "min_topic_size": self.min_topic_size,
                "language_model": self.language_model
            },
            "topic_modeling": topic_results,
            "insights_preparation": insights_data,
            "summary": self._create_summary(topic_results)
        }
        
        return composed_results
    
    def _prepare_insights_data(
        self, 
        df: pd.DataFrame, 
        topic_results: Dict[str, Any],
        sentiment_col: str,
        text_col: str
    ) -> Dict[str, Any]:
        """
        Prepare structured data for LLM insights generation.
        
        Args:
            df: Original DataFrame
            topic_results: Results from topic modeling
            sentiment_col: Sentiment column used
            text_col: Text column used
            
        Returns:
            Dictionary with prepared insights data
        """
        insights_prep = {
            "overview": {
                "app_info": self._extract_app_info(df),
                "review_distribution": self._get_review_distribution(df),
                "sentiment_breakdown": self._get_sentiment_breakdown(df, sentiment_col),
                "time_period": self._get_time_period(df)
            },
            "negative_review_analysis": {
                "total_negative": topic_results["modeling_info"]["negative_reviews"],
                "topics_found": topic_results["modeling_info"]["topics_discovered"],
                "topic_breakdown": []
            },
            "prompts_for_llm": {
                "main_context": "",
                "topic_contexts": {}
            }
        }
        
        # Prepare topic breakdown and LLM prompts
        topics = topic_results["topics"]
        if topics:
            # Sort topics by review count for prioritization
            sorted_topics = sorted(topics.items(), key=lambda x: x[1]['review_count'], reverse=True)
            
            for topic_id, topic_data in sorted_topics:
                topic_breakdown = {
                    "topic_id": topic_id,
                    "topic_name": topic_data["topic_name"],
                    "review_count": topic_data["review_count"],
                    "percentage": topic_data["percentage_of_negative"],
                    "key_phrases": [p["phrase"] for p in topic_data["key_phrases"][:5]],
                    "top_words": [w["word"] for w in topic_data["frequent_words"][:5]],
                    "sample_reviews": topic_data["sample_reviews"][:2]  # Just 2 for context
                }
                insights_prep["negative_review_analysis"]["topic_breakdown"].append(topic_breakdown)
                
                # Prepare topic-specific context for LLM
                topic_context = self._create_topic_context(topic_data)
                insights_prep["prompts_for_llm"]["topic_contexts"][topic_id] = topic_context
            
            # Create main context prompt
            insights_prep["prompts_for_llm"]["main_context"] = self._create_main_context(
                insights_prep["overview"], 
                insights_prep["negative_review_analysis"]
            )
        
        return insights_prep
    
    def _extract_app_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract app information from the DataFrame."""
        app_info = {}
        
        # Try to get app name
        if 'app_name' in df.columns:
            app_info["name"] = df['app_name'].iloc[0] if not df['app_name'].isna().all() else "Unknown App"
        else:
            app_info["name"] = "Unknown App"
        
        # Try to get version info
        if 'version' in df.columns:
            versions = df['version'].dropna().unique()
            app_info["versions_reviewed"] = list(versions)[:5]  # Top 5 versions
        
        # Try to get country info
        if 'country' in df.columns:
            countries = df['country'].dropna().unique()
            app_info["countries"] = list(countries)[:10]  # Top 10 countries
        
        return app_info
    
    def _get_review_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get rating distribution."""
        if 'rating' not in df.columns:
            return {}
        
        rating_counts = df['rating'].value_counts().sort_index()
        total = len(df)
        
        return {
            "total_reviews": total,
            "rating_distribution": {
                str(rating): {
                    "count": int(count),
                    "percentage": round(count / total * 100, 1)
                }
                for rating, count in rating_counts.items()
            },
            "average_rating": round(float(df['rating'].mean()), 2)
        }
    
    def _get_sentiment_breakdown(self, df: pd.DataFrame, sentiment_col: str) -> Dict[str, Any]:
        """Get sentiment distribution if available."""
        if sentiment_col not in df.columns:
            return {}
        
        sentiment_counts = df[sentiment_col].value_counts()
        total = len(df)
        
        return {
            "total_analyzed": total,
            "distribution": {
                sentiment: {
                    "count": int(count),
                    "percentage": round(count / total * 100, 1)
                }
                for sentiment, count in sentiment_counts.items()
            }
        }
    
    def _get_time_period(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get time period of reviews if date column exists."""
        date_columns = ['date', 'review_date', 'created_date', 'timestamp']
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            return {}
        
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(dates) > 0:
                return {
                    "earliest_review": dates.min().strftime("%Y-%m-%d"),
                    "latest_review": dates.max().strftime("%Y-%m-%d"),
                    "date_span_days": (dates.max() - dates.min()).days
                }
        except:
            pass
        
        return {}
    
    def _create_topic_context(self, topic_data: Dict[str, Any]) -> str:
        """Create context string for a specific topic for LLM analysis."""
        context = f"Topic: {topic_data['topic_name']}\n"
        context += f"Reviews in this topic: {topic_data['review_count']} ({topic_data['percentage_of_negative']}% of negative reviews)\n\n"
        
        context += "Key phrases identified:\n"
        for phrase in topic_data['key_phrases'][:5]:
            context += f"- '{phrase['phrase']}' (relevance: {phrase['score']:.3f})\n"
        
        context += "\nMost frequent words:\n"
        for word in topic_data['frequent_words'][:5]:
            context += f"- '{word['word']}' ({word['count']} times)\n"
        
        context += "\nSample reviews from this topic:\n"
        for i, review in enumerate(topic_data['sample_reviews'][:2], 1):
            # Truncate long reviews
            review_text = review if len(review) <= 200 else review[:200] + "..."
            context += f"{i}. \"{review_text}\"\n"
        
        return context
    
    def _create_main_context(self, overview: Dict[str, Any], negative_analysis: Dict[str, Any]) -> str:
        """Create main context for comprehensive LLM analysis."""
        context = "COMPREHENSIVE REVIEW ANALYSIS CONTEXT\n"
        context += "=" * 50 + "\n\n"
        
        # App overview
        app_info = overview.get("app_info", {})
        context += f"App: {app_info.get('name', 'Unknown App')}\n"
        if "countries" in app_info:
            context += f"Markets analyzed: {', '.join(app_info['countries'][:5])}\n"
        
        # Review distribution
        review_dist = overview.get("review_distribution", {})
        if review_dist:
            context += f"Total reviews: {review_dist.get('total_reviews', 0)}\n"
            context += f"Average rating: {review_dist.get('average_rating', 'N/A')}/5\n"
        
        # Sentiment breakdown
        sentiment = overview.get("sentiment_breakdown", {})
        if sentiment:
            dist = sentiment.get("distribution", {})
            context += f"Sentiment distribution: "
            context += ", ".join([f"{s}: {d['percentage']}%" for s, d in dist.items()])
            context += "\n"
        
        context += "\n" + "NEGATIVE REVIEW THEMES DISCOVERED:\n"
        context += f"Total negative reviews analyzed: {negative_analysis['total_negative']}\n"
        context += f"Distinct complaint themes found: {negative_analysis['topics_found']}\n\n"
        
        # Topic summary
        for i, topic in enumerate(negative_analysis['topic_breakdown'], 1):
            context += f"{i}. {topic['topic_name']} ({topic['review_count']} reviews, {topic['percentage']}%)\n"
            context += f"   Key issues: {', '.join(topic['key_phrases'][:3])}\n"
        
        return context
    
    def _create_summary(self, topic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a high-level summary of the analysis."""
        topics = topic_results.get("topics", {})
        modeling_info = topic_results.get("modeling_info", {})
        
        # Find biggest complaint themes
        if topics:
            sorted_topics = sorted(topics.items(), key=lambda x: x[1]['review_count'], reverse=True)
            biggest_themes = [
                {
                    "theme": topic_data["topic_name"],
                    "reviews": topic_data["review_count"],
                    "percentage": topic_data["percentage_of_negative"],
                    "top_issues": [p["phrase"] for p in topic_data["key_phrases"][:3]]
                }
                for topic_id, topic_data in sorted_topics[:3]  # Top 3 themes
            ]
        else:
            biggest_themes = []
        
        return {
            "analysis_scope": {
                "total_reviews": modeling_info.get("total_reviews", 0),
                "negative_reviews": modeling_info.get("negative_reviews", 0),
                "complaint_themes": modeling_info.get("topics_discovered", 0)
            },
            "key_findings": {
                "biggest_complaint_themes": biggest_themes,
                "reviews_categorized": modeling_info.get("negative_reviews", 0) - modeling_info.get("outliers", 0),
                "uncategorized_reviews": modeling_info.get("outliers", 0)
            }
        }
    
    def print_composition_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary of the composed analysis."""
        print("\n" + "="*80)
        print("🎼 REVIEW ANALYSIS COMPOSITION SUMMARY")
        print("="*80)
        
        comp_info = results["composition_info"]
        print(f"Total Reviews: {comp_info['total_reviews']:,}")
        print(f"Negative Reviews: {comp_info['negative_reviews']:,} ({comp_info['negative_reviews']/comp_info['total_reviews']*100:.1f}%)")
        print(f"Topics Discovered: {comp_info['topics_discovered']}")
        print(f"Outliers: {comp_info['outliers']} ({comp_info['outliers']/comp_info['negative_reviews']*100:.1f}%)")
        
        summary = results.get("summary", {})
        key_findings = summary.get("key_findings", {})
        
        if key_findings.get("biggest_complaint_themes"):
            print("\n🎯 TOP COMPLAINT THEMES:")
            print("-" * 40)
            for i, theme in enumerate(key_findings["biggest_complaint_themes"], 1):
                print(f"{i}. {theme['theme']} ({theme['reviews']} reviews, {theme['percentage']}%)")
                print(f"   Key issues: {', '.join(theme['top_issues'])}")
        
        print(f"\n📊 Reviews Categorized: {key_findings.get('reviews_categorized', 0)}")
        print(f"📊 Reviews Uncategorized: {key_findings.get('uncategorized_reviews', 0)}")
        print("\n" + "="*80)


def compose_analysis_csv(
    input_path: Path,
    output_path: Path,
    sentiment_col: str = "review_sentiment_class",
    text_col: str = "review_clean_translated",
    min_topic_size: int = 5,
    language_model: str = "all-MiniLM-L6-v2"
) -> None:
    """
    Compose analysis for a CSV file with sentiment-analyzed review data.
    """
    composer = ReviewComposer(min_topic_size, language_model)
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} reviews from {input_path}")
        
        # Compose analysis
        results = composer.compose_analysis(df, sentiment_col, text_col)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Saved composed analysis to {output_path}")
        
        # Print summary
        composer.print_composition_summary(results)
        
    except Exception as e:
        raise ComposerError(f"Failed to compose analysis for {input_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose comprehensive review analysis with topic modeling.")
    parser.add_argument("input", help="Input CSV file (with sentiment analysis)")
    parser.add_argument("--output", "-o", help="Output JSON file (default: composed_<input>)")
    parser.add_argument("--min-topic-size", type=int, default=5, help="Minimum reviews per topic (default: 5)")
    parser.add_argument("--language-model", default="all-MiniLM-L6-v2", 
                       help="Sentence transformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--sentiment-col", default="review_sentiment_class",
                       help="Column with sentiment classification")
    parser.add_argument("--text-col", default="review_clean_translated",
                       help="Column with review text")
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            raise ComposerError(f"Input file not found: {input_path}")
        
        # Set output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"composed_{input_path.stem}.json"
        
        compose_analysis_csv(
            input_path, 
            output_path, 
            args.sentiment_col,
            args.text_col,
            args.min_topic_size,
            args.language_model
        )
        
    except ComposerError as e:
        print(f"COMPOSER ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(2)