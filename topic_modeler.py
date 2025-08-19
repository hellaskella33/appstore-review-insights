import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import re
from collections import Counter

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import umap


class TopicModelerError(Exception):
    pass


class ReviewTopicModeler:
    """
    BERTopic-based topic modeling with TF-IDF phrase extraction for app reviews.
    Focuses on discovering themes in negative reviews and extracting key phrases.
    """
    
    def __init__(self, min_topic_size: int = 5, language_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the topic modeler.
        
        Args:
            min_topic_size: Minimum number of reviews to form a topic
            language_model: Sentence transformer model for embeddings
        """
        self.min_topic_size = min_topic_size
        self.language_model = language_model
        
        # Initialize components
        self.sentence_model = None
        self.topic_model = None
        self.tfidf_vectorizer = None
        
        # Analysis results
        self.topics_info = {}
        self.stats = {
            "total_reviews": 0,
            "negative_reviews": 0,
            "topics_discovered": 0,
            "outliers": 0
        }
    
    def _initialize_models(self):
        """Initialize the ML models lazily to save memory."""
        if self.sentence_model is None:
            print(f"Loading sentence transformer model: {self.language_model}")
            self.sentence_model = SentenceTransformer(self.language_model)
        
        if self.topic_model is None:
            # Configure BERTopic with optimized settings for app reviews
            umap_model = umap.UMAP(
                n_neighbors=15, 
                n_components=5, 
                min_dist=0.0, 
                metric='cosine',
                random_state=42
            )
            
            self.topic_model = BERTopic(
                embedding_model=self.sentence_model,
                umap_model=umap_model,
                min_topic_size=self.min_topic_size,
                calculate_probabilities=True,
                verbose=False
            )
        
        if self.tfidf_vectorizer is None:
            # TF-IDF for extracting key phrases within topics
            # Custom stop words for app reviews
            custom_stop_words = [
                'app', 'tiktok', 'instagram', 'facebook', 'twitter', 'snapchat',
                'the', 'and', 'to', 'of', 'in', 'a', 'an', 'is', 'it', 'that', 'this',
                'with', 'for', 'on', 'at', 'by', 'from', 'they', 'be', 'or', 'as',
                'but', 'not', 'have', 'had', 'has', 'was', 'were', 'been', 'are',
                'will', 'would', 'could', 'should', 'can', 'my', 'me', 'i', 'you',
                'your', 'we', 'our', 'us', 'he', 'she', 'him', 'her', 'his', 'their',
                'really', 'just', 'like', 'get', 'go', 'know', 'think', 'want', 'need',
                'use', 'used', 'using', 'good', 'bad', 'great', 'terrible', 'awful'
            ]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=500,  # Reduced to focus on most important terms
                ngram_range=(1, 2),  # Unigrams and bigrams only
                stop_words=custom_stop_words,
                max_df=0.7,  # More restrictive
                min_df=3,  # Require at least 3 occurrences
                lowercase=True,
                token_pattern=r'(?u)\b[A-Za-z][A-Za-z]+\b'  # At least 2 letters
            )
    
    def filter_negative_reviews(
        self, 
        df: pd.DataFrame, 
        sentiment_col: str = "review_sentiment_class",
        text_col: str = "review_clean_translated"
    ) -> Tuple[pd.DataFrame, str]:
        """
        Filter for negative reviews with meaningful text.
        
        Args:
            df: DataFrame with reviews and sentiment analysis
            sentiment_col: Column containing sentiment classification
            text_col: Column containing review text to analyze
            
        Returns:
            Tuple of (filtered_dataframe, actual_text_column_used)
        """
        # Check if sentiment column exists
        if sentiment_col not in df.columns:
            # Fallback: use rating if no sentiment analysis
            if 'rating' in df.columns:
                print(f"No sentiment column '{sentiment_col}' found, using rating <= 2 as negative")
                negative_df = df[df['rating'] <= 2].copy()
            else:
                raise TopicModelerError(f"No sentiment column '{sentiment_col}' or 'rating' column found")
        else:
            negative_df = df[df[sentiment_col] == 'negative'].copy()
        
        # Check if text column exists
        if text_col not in df.columns:
            # Try alternative columns
            alternatives = ['review_clean', 'review', 'text_clean_translated', 'text_clean']
            text_col = None
            for alt in alternatives:
                if alt in df.columns:
                    text_col = alt
                    print(f"Using '{text_col}' as text column")
                    break
            
            if text_col is None:
                raise TopicModelerError(f"No suitable text column found. Available: {list(df.columns)}")
        
        # Filter for meaningful text (not empty, not too short)
        negative_df = negative_df[
            negative_df[text_col].notna() & 
            (negative_df[text_col].str.len() >= 20)
        ].copy()
        
        if len(negative_df) == 0:
            raise TopicModelerError("No negative reviews with meaningful text found")
        
        self.stats["total_reviews"] = len(df)
        self.stats["negative_reviews"] = len(negative_df)
        
        print(f"Found {len(negative_df)} negative reviews out of {len(df)} total reviews")
        return negative_df, text_col
    
    def clean_text_for_analysis(self, text: str) -> str:
        """
        Clean text specifically for topic modeling and phrase extraction.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning while preserving meaning
        text = text.lower().strip()
        
        # Remove excessive punctuation but keep some for context
        text = re.sub(r'[^\w\s.,!?\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words that don't add meaning
        words = text.split()
        words = [w for w in words if len(w) >= 2]
        
        return ' '.join(words)
    
    def discover_topics(self, reviews: List[str]) -> Tuple[List[int], List[float]]:
        """
        Discover topics in negative reviews using BERTopic.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Tuple of (topic_assignments, probabilities)
        """
        self._initialize_models()
        
        print(f"Discovering topics in {len(reviews)} negative reviews...")
        
        # Clean texts for analysis
        cleaned_reviews = [self.clean_text_for_analysis(review) for review in reviews]
        
        # Remove empty reviews
        non_empty_reviews = [(i, review) for i, review in enumerate(cleaned_reviews) if len(review.strip()) > 10]
        
        if len(non_empty_reviews) < self.min_topic_size:
            raise TopicModelerError(f"Not enough reviews ({len(non_empty_reviews)}) for topic discovery (need at least {self.min_topic_size})")
        
        indices, clean_texts = zip(*non_empty_reviews)
        
        # Fit BERTopic
        topics, probabilities = self.topic_model.fit_transform(list(clean_texts))
        
        # Create full results array (including empty reviews as outliers)
        full_topics = [-1] * len(reviews)  # -1 = outlier
        full_probabilities = [0.0] * len(reviews)
        
        for i, (orig_idx, topic, prob) in enumerate(zip(indices, topics, probabilities)):
            full_topics[orig_idx] = topic
            full_probabilities[orig_idx] = prob if isinstance(prob, (int, float)) else prob[0] if hasattr(prob, '__len__') else 0.0
        
        # Get topic info
        self.topics_info = self.topic_model.get_topic_info()
        self.stats["topics_discovered"] = len(self.topics_info) - 1  # Exclude outlier topic
        self.stats["outliers"] = sum(1 for t in full_topics if t == -1)
        
        print(f"Discovered {self.stats['topics_discovered']} topics")
        print(f"Outliers (uncategorized): {self.stats['outliers']} reviews")
        
        return full_topics, full_probabilities
    
    def extract_phrases_per_topic(self, df_negative: pd.DataFrame, text_col: str, topics: List[int]) -> Dict[int, Dict]:
        """
        Extract key phrases for each topic using TF-IDF.
        
        Args:
            df_negative: DataFrame with negative reviews
            text_col: Column name with review text
            topics: Topic assignments for each review
            
        Returns:
            Dictionary with phrases per topic
        """
        phrases_per_topic = {}
        
        # Group reviews by topic
        df_with_topics = df_negative.copy()
        df_with_topics['topic'] = topics
        
        topic_groups = df_with_topics.groupby('topic')
        
        for topic_id, group in topic_groups:
            if topic_id == -1:  # Skip outliers
                continue
            
            topic_texts = group[text_col].tolist()
            cleaned_texts = [self.clean_text_for_analysis(text) for text in topic_texts]
            cleaned_texts = [text for text in cleaned_texts if len(text.strip()) > 10]
            
            if len(cleaned_texts) < 2:
                continue
            
            try:
                # Fit TF-IDF on this topic's texts
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
                # Get mean TF-IDF scores for each term
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Get top phrases
                top_indices = np.argsort(mean_scores)[::-1][:15]  # Top 15 phrases
                top_phrases = [
                    {
                        "phrase": feature_names[idx],
                        "score": float(mean_scores[idx]),
                        "length": len(feature_names[idx].split())
                    }
                    for idx in top_indices if mean_scores[idx] > 0.01  # Filter very low scores
                ]
                
                # Also get word frequency for context
                all_words = ' '.join(cleaned_texts).split()
                word_freq = Counter(all_words)
                top_words = [{"word": word, "count": count} for word, count in word_freq.most_common(10)]
                
                phrases_per_topic[topic_id] = {
                    "review_count": len(group),
                    "top_phrases": top_phrases,
                    "top_words": top_words,
                    "sample_reviews": topic_texts[:3]  # First 3 reviews as examples
                }
                
            except Exception as e:
                print(f"Warning: Could not extract phrases for topic {topic_id}: {e}")
                continue
        
        return phrases_per_topic
    
    def get_topic_names(self) -> Dict[int, str]:
        """
        Generate meaningful names for topics based on their content.
        """
        topic_names = {}
        
        if self.topic_model is None:
            return topic_names
        
        # Get topic words from BERTopic
        topics = self.topic_model.get_topics()
        
        # Common stop words and function words to filter out
        stop_words = {
            'the', 'and', 'to', 'of', 'in', 'a', 'an', 'is', 'it', 'that', 'this',
            'with', 'for', 'on', 'at', 'by', 'from', 'they', 'be', 'or', 'as',
            'but', 'not', 'have', 'had', 'has', 'was', 'were', 'been', 'are',
            'will', 'would', 'could', 'should', 'can', 'my', 'me', 'i', 'you',
            'your', 'we', 'our', 'us', 'he', 'she', 'him', 'her', 'his', 'their'
        }
        
        for topic_id, words in topics.items():
            if topic_id == -1:  # Skip outlier topic
                continue
            
            # Filter out stop words and get meaningful words
            meaningful_words = []
            for word, _ in words[:10]:  # Look at top 10 words
                if (word.lower() not in stop_words and 
                    len(word) > 2 and 
                    word.isalpha()):  # Only alphabetic words
                    meaningful_words.append(word)
                
                # Stop when we have 3 meaningful words
                if len(meaningful_words) >= 3:
                    break
            
            # If we don't have enough meaningful words, use all available
            if len(meaningful_words) == 0:
                meaningful_words = [word for word, _ in words[:3] if len(word) > 1]
            
            # Create topic name
            if meaningful_words:
                topic_name = " + ".join(meaningful_words)
            else:
                topic_name = f"Topic {topic_id}"
            
            topic_names[topic_id] = topic_name
        
        return topic_names
    
    def model_topics(self, df: pd.DataFrame, sentiment_col: str = "review_sentiment_class", text_col: str = "review_clean_translated") -> Dict[str, Any]:
        """
        Main method to perform topic modeling and phrase extraction.
        
        Args:
            df: DataFrame with review data including sentiment analysis
            sentiment_col: Column containing sentiment classification
            text_col: Column containing review text to analyze
            
        Returns:
            Dictionary with complete topic modeling results
        """
        # Filter negative reviews
        df_negative, actual_text_col = self.filter_negative_reviews(df, sentiment_col, text_col)
        
        if len(df_negative) < self.min_topic_size:
            raise TopicModelerError(f"Need at least {self.min_topic_size} negative reviews for analysis, found {len(df_negative)}")
        
        # Discover topics
        reviews_list = df_negative[actual_text_col].tolist()
        topics, probabilities = self.discover_topics(reviews_list)
        
        # Extract phrases per topic
        phrases_per_topic = self.extract_phrases_per_topic(df_negative, actual_text_col, topics)
        
        # Get topic names
        topic_names = self.get_topic_names()
        
        # Handle BERTopic representation safely
        def safe_get_bertopic_words(topic_id):
            topic_info = self.topics_info[self.topics_info['Topic'] == topic_id].iloc[0] if len(self.topics_info) > topic_id + 1 else None
            bertopic_words = []
            if topic_info is not None and 'Representation' in topic_info:
                representation = topic_info['Representation']
                if isinstance(representation, str):
                    bertopic_words = representation.split(', ')
                elif isinstance(representation, list):
                    bertopic_words = [str(word) for word in representation]
            return bertopic_words
        
        # Build comprehensive results
        results = {
            "modeling_info": {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "total_reviews": self.stats["total_reviews"],
                "negative_reviews": self.stats["negative_reviews"],
                "topics_discovered": self.stats["topics_discovered"],
                "outliers": self.stats["outliers"],
                "text_column_used": actual_text_col,
                "sentiment_column_used": sentiment_col,
                "min_topic_size": self.min_topic_size,
                "language_model": self.language_model
            },
            "topics": {},
            "raw_data": {
                "topic_assignments": topics,
                "topic_probabilities": probabilities,
                "negative_review_indices": df_negative.index.tolist()
            }
        }
        
        # Populate topic results
        for topic_id, phrase_data in phrases_per_topic.items():
            results["topics"][topic_id] = {
                "topic_name": topic_names.get(topic_id, f"Topic {topic_id}"),
                "review_count": phrase_data["review_count"],
                "percentage_of_negative": round(phrase_data["review_count"] / self.stats["negative_reviews"] * 100, 1),
                "bertopic_words": safe_get_bertopic_words(topic_id),
                "key_phrases": phrase_data["top_phrases"],
                "frequent_words": phrase_data["top_words"],
                "sample_reviews": phrase_data["sample_reviews"]
            }
        
        return results


def model_topics_csv(
    input_path: Path,
    output_path: Path,
    sentiment_col: str = "review_sentiment_class",
    text_col: str = "review_clean_translated",
    min_topic_size: int = 5,
    language_model: str = "all-MiniLM-L6-v2"
) -> None:
    """
    Perform topic modeling on a CSV file with sentiment-analyzed review data.
    """
    modeler = ReviewTopicModeler(min_topic_size, language_model)
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} reviews from {input_path}")
        
        # Model topics
        results = modeler.model_topics(df, sentiment_col, text_col)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Saved topic modeling results to {output_path}")
        
        # Print summary
        info = results["modeling_info"]
        print(f"\n📊 TOPIC MODELING SUMMARY")
        print(f"Total Reviews: {info['total_reviews']:,}")
        print(f"Negative Reviews: {info['negative_reviews']:,}")
        print(f"Topics Discovered: {info['topics_discovered']}")
        print(f"Outliers: {info['outliers']}")
        
        # Show topics
        topics = results["topics"]
        if topics:
            print("\n🎯 DISCOVERED TOPICS:")
            for topic_id, topic_data in sorted(topics.items(), key=lambda x: x[1]['review_count'], reverse=True):
                print(f"  Topic {topic_id}: {topic_data['topic_name']} ({topic_data['review_count']} reviews)")
        
    except Exception as e:
        raise TopicModelerError(f"Failed to model topics for {input_path}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Perform BERTopic modeling and TF-IDF phrase extraction on app reviews.")
    parser.add_argument("input", help="Input CSV file (with sentiment analysis)")
    parser.add_argument("--output", "-o", help="Output JSON file (default: topics_<input>)")
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
            raise TopicModelerError(f"Input file not found: {input_path}")
        
        # Set output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"topics_{input_path.stem}.json"
        
        model_topics_csv(
            input_path, 
            output_path, 
            args.sentiment_col,
            args.text_col,
            args.min_topic_size,
            args.language_model
        )
        
    except TopicModelerError as e:
        print(f"TOPIC MODELING ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(2)