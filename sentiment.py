import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentError(Exception):
    pass


class SentimentAnalyzer:
    """
    Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Designed specifically for social media text and app reviews.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stats = {
            "total_reviews": 0,
            "analyzed_reviews": 0,
            "sentiment_distribution": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            },
            "avg_scores": {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
                "compound": 0.0
            }
        }
    
    def clean_text_for_sentiment(self, text: str) -> str:
        """
        Clean text for optimal VADER analysis while preserving sentiment indicators.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # VADER handles emojis, punctuation, and caps well, so minimal cleaning
        text = text.strip()
        
        # Remove excessive whitespace but keep punctuation for emphasis
        text = re.sub(r'\s+', ' ', text)
        
        # Keep emojis, exclamation marks, caps - they're sentiment indicators
        return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores:
            - positive: 0.0 to 1.0
            - negative: 0.0 to 1.0  
            - neutral: 0.0 to 1.0
            - compound: -1.0 to 1.0 (overall sentiment)
        """
        if not text or not isinstance(text, str):
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "compound": 0.0
            }
        
        cleaned_text = self.clean_text_for_sentiment(text)
        if not cleaned_text:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "compound": 0.0
            }
        
        # VADER returns scores as dict
        scores = self.analyzer.polarity_scores(cleaned_text)
        
        return {
            "positive": scores['pos'],
            "negative": scores['neg'],
            "neutral": scores['neu'],
            "compound": scores['compound']
        }
    
    def classify_sentiment(self, compound_score: float, threshold: float = 0.05) -> str:
        """
        Classify sentiment based on compound score.
        
        Args:
            compound_score: VADER compound score (-1.0 to 1.0)
            threshold: Threshold for neutral classification
            
        Returns:
            'positive', 'negative', or 'neutral'
        """
        if compound_score >= threshold:
            return "positive"
        elif compound_score <= -threshold:
            return "negative"
        else:
            return "neutral"
    
    def get_sentiment_intensity(self, compound_score: float) -> str:
        """
        Get sentiment intensity level.
        
        Args:
            compound_score: VADER compound score (-1.0 to 1.0)
            
        Returns:
            Intensity level string
        """
        abs_score = abs(compound_score)
        
        if abs_score >= 0.8:
            return "very_strong"
        elif abs_score >= 0.6:
            return "strong"
        elif abs_score >= 0.3:
            return "moderate"
        elif abs_score >= 0.1:
            return "weak"
        else:
            return "neutral"
    
    def analyze_review_data(self, df: pd.DataFrame, text_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze sentiment for review data in DataFrame.
        
        Args:
            df: DataFrame with review data (should have translated columns)
            text_columns: Columns to analyze (default: looks for translated columns)
            
        Returns:
            DataFrame with sentiment analysis columns added
        """
        if text_columns is None:
            # Look for translated columns first, fallback to clean columns
            text_columns = []
            if 'title_clean_translated' in df.columns:
                text_columns.append('title_clean_translated')
            elif 'title_clean' in df.columns:
                text_columns.append('title_clean')
            
            if 'review_clean_translated' in df.columns:
                text_columns.append('review_clean_translated')
            elif 'review_clean' in df.columns:
                text_columns.append('review_clean')
        
        if not text_columns:
            raise SentimentError("No suitable text columns found. Run processor.py and/or translation.py first.")
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in df.columns]
        if missing_cols:
            raise SentimentError(f"Missing columns in DataFrame: {missing_cols}")
        
        df_sentiment = df.copy()
        self.stats["total_reviews"] = len(df)
        
        # Track cumulative scores for averages
        total_scores = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
            "compound": 0.0
        }
        
        for col in text_columns:
            print(f"Analyzing sentiment for column: {col}")
            
            # Add new columns for sentiment analysis
            base_name = col.replace('_clean_translated', '').replace('_clean', '')
            sentiment_cols = {
                'positive': f"{base_name}_sentiment_positive",
                'negative': f"{base_name}_sentiment_negative", 
                'neutral': f"{base_name}_sentiment_neutral",
                'compound': f"{base_name}_sentiment_compound",
                'classification': f"{base_name}_sentiment_class",
                'intensity': f"{base_name}_sentiment_intensity"
            }
            
            # Initialize columns
            for key, col_name in sentiment_cols.items():
                if key in ['positive', 'negative', 'neutral', 'compound']:
                    df_sentiment[col_name] = 0.0
                else:
                    df_sentiment[col_name] = ""
            
            for idx, row in df_sentiment.iterrows():
                text = str(row[col]) if pd.notna(row[col]) else ""
                
                if len(text.strip()) == 0:
                    continue
                
                # Analyze sentiment
                try:
                    sentiment_scores = self.analyze_sentiment(text)
                    
                    # Add scores to DataFrame
                    df_sentiment.at[idx, sentiment_cols['positive']] = sentiment_scores['positive']
                    df_sentiment.at[idx, sentiment_cols['negative']] = sentiment_scores['negative']
                    df_sentiment.at[idx, sentiment_cols['neutral']] = sentiment_scores['neutral']
                    df_sentiment.at[idx, sentiment_cols['compound']] = sentiment_scores['compound']
                    
                    # Add classification and intensity
                    classification = self.classify_sentiment(sentiment_scores['compound'])
                    intensity = self.get_sentiment_intensity(sentiment_scores['compound'])
                    
                    df_sentiment.at[idx, sentiment_cols['classification']] = classification
                    df_sentiment.at[idx, sentiment_cols['intensity']] = intensity
                    
                    # Update stats
                    self.stats["sentiment_distribution"][classification] += 1
                    for score_type, score_value in sentiment_scores.items():
                        total_scores[score_type] += score_value
                    
                    self.stats["analyzed_reviews"] += 1
                    
                    if idx % 50 == 0 and idx > 0:  # Progress indicator
                        print(f"  Processed {idx}/{len(df)} reviews...")
                        
                except Exception as e:
                    print(f"Sentiment analysis error for row {idx}: {e}")
                    continue
        
        # Calculate average scores
        if self.stats["analyzed_reviews"] > 0:
            for score_type in total_scores:
                self.stats["avg_scores"][score_type] = round(
                    total_scores[score_type] / self.stats["analyzed_reviews"], 3
                )
        
        return df_sentiment
    
    def print_stats(self):
        """Print sentiment analysis statistics."""
        print("\n=== SENTIMENT ANALYSIS STATISTICS ===")
        print(f"Total reviews processed: {self.stats['total_reviews']}")
        print(f"Reviews analyzed: {self.stats['analyzed_reviews']}")
        
        if self.stats['analyzed_reviews'] > 0:
            dist = self.stats['sentiment_distribution']
            total = sum(dist.values())
            
            print("\nSentiment Distribution:")
            print(f"  👍 Positive: {dist['positive']} ({dist['positive']/total*100:.1f}%)")
            print(f"  😐 Neutral:  {dist['negative']} ({dist['negative']/total*100:.1f}%)")
            print(f"  👎 Negative: {dist['neutral']} ({dist['neutral']/total*100:.1f}%)")
            
            avg = self.stats['avg_scores']
            print(f"\nAverage VADER Scores:")
            print(f"  Positive: {avg['positive']:.3f}")
            print(f"  Negative: {avg['negative']:.3f}")
            print(f"  Neutral:  {avg['neutral']:.3f}")
            print(f"  Compound: {avg['compound']:.3f}")
    
    def get_sentiment_summary(self) -> Dict:
        """Get sentiment analysis summary for export."""
        return {
            "total_reviews": self.stats["total_reviews"],
            "analyzed_reviews": self.stats["analyzed_reviews"],
            "sentiment_distribution": self.stats["sentiment_distribution"].copy(),
            "average_scores": self.stats["avg_scores"].copy(),
            "overall_sentiment": self.classify_sentiment(self.stats["avg_scores"]["compound"])
        }


def analyze_sentiment_csv(
    input_path: Path,
    output_path: Path,
    text_columns: Optional[List[str]] = None
) -> None:
    """
    Analyze sentiment for a CSV file with review data.
    """
    analyzer = SentimentAnalyzer()
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} reviews from {input_path}")
        
        # Analyze sentiment
        df_sentiment = analyzer.analyze_review_data(df, text_columns)
        
        # Save sentiment analysis data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_sentiment.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        
        # Save sentiment metadata
        meta_path = output_path.with_suffix('.sentiment.json')
        sentiment_summary = analyzer.get_sentiment_summary()
        
        metadata = {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(input_path),
            "total_reviews": len(df),
            "analyzed_reviews": analyzer.stats["analyzed_reviews"],
            "sentiment_summary": sentiment_summary,
            "columns_analyzed": text_columns,
            "vader_version": "3.3.2",
            "new_columns_added": [col for col in df_sentiment.columns if col not in df.columns]
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved sentiment data to {output_path}")
        print(f"Saved metadata to {meta_path}")
        
        analyzer.print_stats()
        
    except Exception as e:
        raise SentimentError(f"Failed to analyze sentiment for {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment of app review data using VADER.")
    parser.add_argument("input", help="Input CSV file (translated/processed reviews)")
    parser.add_argument("--output", "-o", help="Output CSV file (default: sentiment_<input>)")
    parser.add_argument("--columns", nargs="+", help="Text columns to analyze (default: auto-detect translated columns)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Neutral sentiment threshold (default: 0.05)")
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            raise SentimentError(f"Input file not found: {input_path}")
        
        # Set output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"sentiment_{input_path.name}"
        
        # Set columns to analyze
        text_columns = args.columns
        
        analyze_sentiment_csv(input_path, output_path, text_columns)
        
    except SentimentError as e:
        print(f"SENTIMENT ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()