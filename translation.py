import argparse
import csv
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TranslationError(Exception):
    pass


class LanguageDetector:
    """
    Language detection using langdetect library.
    """
    
    def __init__(self):
        # Set random seed for consistent results
        DetectorFactory.seed = 0
    
    def detect_language(self, text: str, confidence_threshold: float = 0.5) -> Tuple[str, float]:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence score
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or not isinstance(text, str):
            return "unknown", 0.0
        
        # Clean text for detection
        text = text.replace('\n', ' ').replace('\r', ' ').strip()
        if len(text) < 5:  # Need at least 5 characters for reliable detection
            return "unknown", 0.0
        
        try:
            # langdetect is quite reliable, but doesn't provide confidence
            # We'll simulate confidence based on text length and language
            lang_code = detect(text)
            
            # Simulate confidence based on text length and common languages
            confidence = 0.95 if len(text) > 20 else 0.80
            if lang_code in ['en', 'es', 'fr', 'de', 'it', 'pt']:
                confidence += 0.05  # Boost for common languages
            
            return lang_code, confidence
            
        except LangDetectException:
            return "unknown", 0.0
        except Exception as e:
            print(f"Warning: Language detection failed: {e}")
            return "unknown", 0.0


class Translator:
    """
    Translation service using OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPEN_AI_API_KEY")
        if not self.api_key:
            raise TranslationError("OpenAI API key not found. Set OPEN_AI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.translation_cache = {}  # Simple in-memory cache
        self.request_count = 0
        self.last_request_time = 0
    
    def _rate_limit(self, requests_per_minute: int = 50):
        """Simple rate limiting."""
        current_time = time.time()
        if current_time - self.last_request_time < 60 / requests_per_minute:
            sleep_time = (60 / requests_per_minute) - (current_time - self.last_request_time)
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def translate_text(
        self, 
        text: str, 
        target_language: str = "English",
        source_language: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Translate text using OpenAI API.
        
        Args:
            text: Text to translate
            target_language: Target language name (e.g., "English", "Spanish")
            source_language: Source language name (optional)
            use_cache: Whether to use translation cache
            
        Returns:
            Translated text
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        if len(text) == 0:
            return ""
        
        # Check cache first
        cache_key = f"{text[:100]}_{target_language}"
        if use_cache and cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Prepare prompt
            if source_language:
                prompt = f"Translate the following text from {source_language} to {target_language}. Only return the translated text, no explanations:\n\n{text}"
            else:
                prompt = f"Translate the following text to {target_language}. Only return the translated text, no explanations:\n\n{text}"
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate texts accurately while preserving the original meaning and tone."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Cache the result
            if use_cache:
                self.translation_cache[cache_key] = translated_text
            
            self.request_count += 1
            return translated_text
            
        except Exception as e:
            print(f"Warning: Translation failed for text: {e}")
            return text  # Return original text if translation fails


class ReviewTranslator:
    """
    Main class for translating app review data.
    """
    
    def __init__(self, target_language: str = "English"):
        self.detector = LanguageDetector()
        self.translator = Translator()
        self.target_language = target_language
        self.stats = {
            "total_reviews": 0,
            "translated_reviews": 0,
            "languages_detected": {},
            "translation_errors": 0
        }
    
    def should_translate(self, lang_code: str, confidence: float) -> bool:
        """
        Determine if text should be translated based on language and confidence.
        """
        # Don't translate if already in English or confidence too low
        if lang_code in ['en', 'unknown'] or confidence < 0.7:
            return False
        return True
    
    def translate_review_data(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """
        Translate review data in DataFrame.
        
        Args:
            df: DataFrame with review data
            text_columns: Columns to translate (default: ['title_clean', 'review_clean'])
            
        Returns:
            DataFrame with translated columns added
        """
        if text_columns is None:
            text_columns = ['title_clean', 'review_clean']
        
        # Validate columns exist
        missing_cols = [col for col in text_columns if col not in df.columns]
        if missing_cols:
            raise TranslationError(f"Missing columns in DataFrame: {missing_cols}")
        
        df_translated = df.copy()
        self.stats["total_reviews"] = len(df)
        
        for col in text_columns:
            print(f"Processing column: {col}")
            
            # Add new columns for translations
            lang_col = f"{col}_language"
            conf_col = f"{col}_language_confidence"
            trans_col = f"{col}_translated"
            
            df_translated[lang_col] = ""
            df_translated[conf_col] = 0.0
            df_translated[trans_col] = ""
            
            for idx, row in df_translated.iterrows():
                text = str(row[col]) if pd.notna(row[col]) else ""
                
                if len(text.strip()) == 0:
                    continue
                
                # Detect language
                lang_code, confidence = self.detector.detect_language(text)
                df_translated.at[idx, lang_col] = lang_code
                df_translated.at[idx, conf_col] = confidence
                
                # Update stats
                if lang_code not in self.stats["languages_detected"]:
                    self.stats["languages_detected"][lang_code] = 0
                self.stats["languages_detected"][lang_code] += 1
                
                # Translate if needed
                if self.should_translate(lang_code, confidence):
                    try:
                        translated = self.translator.translate_text(
                            text, 
                            self.target_language,
                            source_language=self._get_language_name(lang_code)
                        )
                        df_translated.at[idx, trans_col] = translated
                        self.stats["translated_reviews"] += 1
                        
                        if idx % 10 == 0:  # Progress indicator
                            print(f"  Processed {idx + 1}/{len(df)} reviews...")
                            
                    except Exception as e:
                        print(f"Translation error for row {idx}: {e}")
                        df_translated.at[idx, trans_col] = text  # Keep original
                        self.stats["translation_errors"] += 1
                else:
                    # Keep original text
                    df_translated.at[idx, trans_col] = text
        
        return df_translated
    
    def _get_language_name(self, lang_code: str) -> str:
        """Convert language code to full name."""
        lang_map = {
            'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
            'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean',
            'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi', 'tr': 'Turkish',
            'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish',
            'no': 'Norwegian', 'fi': 'Finnish', 'el': 'Greek', 'he': 'Hebrew'
        }
        return lang_map.get(lang_code, lang_code)
    
    def print_stats(self):
        """Print translation statistics."""
        print("\n=== TRANSLATION STATISTICS ===")
        print(f"Total reviews processed: {self.stats['total_reviews']}")
        print(f"Reviews translated: {self.stats['translated_reviews']}")
        print(f"Translation errors: {self.stats['translation_errors']}")
        print(f"API requests made: {self.translator.request_count}")
        print("\nLanguages detected:")
        
        for lang, count in sorted(self.stats['languages_detected'].items(), 
                                 key=lambda x: x[1], reverse=True):
            lang_name = self._get_language_name(lang)
            pct = (count / self.stats['total_reviews']) * 100
            print(f"  {lang_name} ({lang}): {count} reviews ({pct:.1f}%)")


def translate_csv_file(
    input_path: Path, 
    output_path: Path,
    target_language: str = "English",
    text_columns: Optional[List[str]] = None
) -> None:
    """
    Translate a CSV file with review data.
    """
    translator = ReviewTranslator(target_language)
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} reviews from {input_path}")
        
        # Translate
        df_translated = translator.translate_review_data(df, text_columns)
        
        # Save translated data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_translated.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        
        # Save translation metadata
        meta_path = output_path.with_suffix('.translation.json')
        metadata = {
            "translated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(input_path),
            "target_language": target_language,
            "total_reviews": len(df),
            "translated_reviews": translator.stats["translated_reviews"],
            "languages_detected": translator.stats["languages_detected"],
            "translation_errors": translator.stats["translation_errors"],
            "api_requests": translator.translator.request_count,
            "columns_translated": text_columns or ['title_clean', 'review_clean']
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved translated data to {output_path}")
        print(f"Saved metadata to {meta_path}")
        
        translator.print_stats()
        
    except Exception as e:
        raise TranslationError(f"Failed to translate {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Translate app review data using language detection and OpenAI API.")
    parser.add_argument("input", help="Input CSV file (cleaned/processed reviews)")
    parser.add_argument("--output", "-o", help="Output CSV file (default: translated_<input>)")
    parser.add_argument("--language", default="English", help="Target language for translation")
    parser.add_argument("--columns", nargs="+", help="Columns to translate (default: title_clean review_clean)")
    parser.add_argument("--dry-run", action="store_true", help="Only detect languages, don't translate")
    
    args = parser.parse_args()
    
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            raise TranslationError(f"Input file not found: {input_path}")
        
        # Set output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"translated_{input_path.name}"
        
        # Set columns to translate
        text_columns = args.columns or ['title_clean', 'review_clean']
        
        if args.dry_run:
            # Only detect languages
            print("DRY RUN: Detecting languages only...")
            df = pd.read_csv(input_path)
            detector = LanguageDetector()
            
            language_stats = {}
            for col in text_columns:
                if col in df.columns:
                    for text in df[col].fillna(""):
                        if text.strip():
                            lang, conf = detector.detect_language(str(text))
                            if lang not in language_stats:
                                language_stats[lang] = 0
                            language_stats[lang] += 1
            
            print("\nLanguage detection results:")
            for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(df)) * 100
                print(f"  {lang}: {count} texts ({pct:.1f}%)")
        else:
            # Full translation
            translate_csv_file(input_path, output_path, args.language, text_columns)
        
    except TranslationError as e:
        print(f"TRANSLATION ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()