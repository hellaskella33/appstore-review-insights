import argparse
import csv
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
import string

import pandas as pd


class ProcessorError(Exception):
    pass


class TextProcessor:
    """
    Text cleaning and preprocessing utilities.
    Focused purely on cleaning and normalization - no statistics or analysis.
    """
    
    def __init__(self):
        # Common contractions mapping
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
            "i've": "i have", "isn't": "is not", "it'd": "it would",
            "it'll": "it will", "it's": "it is", "let's": "let us",
            "shouldn't": "should not", "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will", "they're": "they are",
            "they've": "they have", "we'd": "we would", "we're": "we are",
            "we've": "we have", "weren't": "were not", "what's": "what is",
            "where's": "where is", "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
        
        # Common app store abbreviations and slang
        self.app_abbreviations = {
            "app": "application", "apps": "applications", "ui": "user interface",
            "ux": "user experience", "bug": "software bug", "lag": "slow performance",
            "glitch": "software glitch", "fix": "repair", "update": "software update",
            "dev": "developer", "devs": "developers"
        }
    
    def clean_text(self, text: str, normalize: bool = False) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            normalize: If True, applies normalization (lowercase, contractions, etc.)
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove null bytes and control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common unicode issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('…', '...')
        
        # Remove excessive punctuation (3+ consecutive)
        text = re.sub(r'([.!?]){3,}', r'\1\1\1', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.!?:;])\s*', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        if normalize:
            # Convert to lowercase
            text = text.lower()
            
            # Expand contractions
            words = text.split()
            expanded_words = []
            for word in words:
                # Remove punctuation for contraction lookup
                clean_word = word.strip(string.punctuation).lower()
                if clean_word in self.contractions:
                    expanded_words.append(self.contractions[clean_word])
                else:
                    expanded_words.append(word)
            text = ' '.join(expanded_words)
            
            # Expand app-specific abbreviations
            words = text.split()
            expanded_words = []
            for word in words:
                clean_word = word.strip(string.punctuation).lower()
                if clean_word in self.app_abbreviations:
                    expanded_words.append(self.app_abbreviations[clean_word])
                else:
                    expanded_words.append(word)
            text = ' '.join(expanded_words)
            
            # Remove excessive punctuation and special characters
            text = re.sub(r'[^\w\s.,!?-]', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
        
        return text


class ReviewProcessor:
    """
    Main processor for app store reviews data.
    Focused on cleaning, normalization, and deduplication only.
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.required_columns = {"userName", "title", "rating", "review"}
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that DataFrame has required columns."""
        if df.empty:
            raise ProcessorError("DataFrame is empty")
        
        missing_cols = self.required_columns - set(df.columns)
        if missing_cols:
            raise ProcessorError(f"Missing required columns: {missing_cols}")
    
    def clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize text fields. Add length info but no calculations.
        """
        self.validate_dataframe(df)
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text fields (preserve readability)
        processed_df['title_clean'] = processed_df['title'].fillna('').astype(str).apply(
            lambda x: self.text_processor.clean_text(x, normalize=False)
        )
        processed_df['review_clean'] = processed_df['review'].fillna('').astype(str).apply(
            lambda x: self.text_processor.clean_text(x, normalize=False)
        )
        
        # Normalized text fields (for analysis)
        processed_df['title_normalized'] = processed_df['title'].fillna('').astype(str).apply(
            lambda x: self.text_processor.clean_text(x, normalize=True)
        )
        processed_df['review_normalized'] = processed_df['review'].fillna('').astype(str).apply(
            lambda x: self.text_processor.clean_text(x, normalize=True)
        )
        
        # Combined text fields
        processed_df['text_clean'] = (
            processed_df['title_clean'].fillna('') + ' ' + 
            processed_df['review_clean'].fillna('')
        ).str.strip()
        
        processed_df['text_normalized'] = (
            processed_df['title_normalized'].fillna('') + ' ' + 
            processed_df['review_normalized'].fillna('')
        ).str.strip()
        
        # Basic length info (no calculations, just counts)
        processed_df['text_length'] = processed_df['text_clean'].str.len()
        processed_df['word_count'] = processed_df['text_clean'].str.split().str.len()
        
        # Content flags
        processed_df['has_title'] = processed_df['title_clean'].str.len() > 0
        processed_df['has_review'] = processed_df['review_clean'].str.len() > 0
        
        # Clean other fields
        processed_df['rating'] = pd.to_numeric(processed_df['rating'], errors='coerce')
        processed_df['rating'] = processed_df['rating'].clip(lower=1, upper=5)
        
        # Clean username
        processed_df['username_clean'] = processed_df['userName'].fillna('Anonymous').astype(str).apply(
            lambda x: self.text_processor.clean_text(x, normalize=False)
        )
        
        # Date processing
        if 'date' in processed_df.columns:
            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
        
        return processed_df
    
    def deduplicate_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate reviews based on normalized text.
        """
        if 'text_normalized' not in df.columns:
            raise ProcessorError("DataFrame must be processed first (missing text_normalized column)")
        
        original_count = len(df)
        
        # Remove exact duplicates based on normalized text
        deduplicated_df = df.drop_duplicates(
            subset=['text_normalized'], 
            keep='first'
        )
        
        # Remove very short/empty reviews
        deduplicated_df = deduplicated_df[
            (deduplicated_df['text_length'] >= 5) |
            (deduplicated_df['has_title'] & deduplicated_df['has_review'])
        ]
        
        filtered_count = len(deduplicated_df)
        print(f"Removed {original_count - filtered_count} duplicates/empty reviews, kept {filtered_count}")
        
        return deduplicated_df


def process_csv_file(
    input_path: Path, 
    output_path: Path,
    skip_dedup: bool = False
) -> None:
    """
    Process a single CSV file - clean text and deduplicate.
    """
    processor = ReviewProcessor()
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} reviews from {input_path}")
        
        # Clean and normalize
        processed_df = processor.clean_and_normalize(df)
        
        # Deduplicate if requested
        if not skip_dedup:
            processed_df = processor.deduplicate_reviews(processed_df)
        
        # Save processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        
        # Create simple metadata
        meta_path = output_path.with_suffix('.meta.json')
        metadata = {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(input_path),
            "original_count": len(df),
            "processed_count": len(processed_df),
            "fields_added": [
                "title_clean", "review_clean", "title_normalized", "review_normalized",
                "text_clean", "text_normalized", "text_length", "word_count",
                "has_title", "has_review", "username_clean"
            ],
            "deduplicated": not skip_dedup
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved processed data to {output_path}")
        print(f"Saved metadata to {meta_path}")
        
    except Exception as e:
        raise ProcessorError(f"Failed to process {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Clean and normalize App Store review data.")
    parser.add_argument("input", help="Input CSV file or directory containing CSV files")
    parser.add_argument("--output", "-o", help="Output file or directory (default: cleaned_<input>)")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--pattern", default="*.csv", help="File pattern for directory processing")
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input)
        
        if not input_path.exists():
            raise ProcessorError(f"Input path does not exist: {input_path}")
        
        if input_path.is_file():
            # Process single file
            if not args.output:
                output_path = input_path.parent / f"cleaned_{input_path.name}"
            else:
                output_path = Path(args.output)
            
            process_csv_file(
                input_path=input_path,
                output_path=output_path,
                skip_dedup=args.skip_dedup
            )
            
        elif input_path.is_dir():
            # Process directory
            output_dir = Path(args.output) if args.output else input_path / "cleaned"
            csv_files = list(input_path.glob(args.pattern))
            
            if not csv_files:
                raise ProcessorError(f"No CSV files found in {input_path} matching {args.pattern}")
            
            print(f"Found {len(csv_files)} files to process")
            
            for csv_file in csv_files:
                output_file = output_dir / f"cleaned_{csv_file.name}"
                try:
                    process_csv_file(
                        input_path=csv_file,
                        output_path=output_file,
                        skip_dedup=args.skip_dedup
                    )
                except ProcessorError as e:
                    print(f"Error processing {csv_file}: {e}", file=sys.stderr)
                    continue
        else:
            raise ProcessorError(f"Input path is neither file nor directory: {input_path}")
            
    except ProcessorError as e:
        print(f"PROCESSOR ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()