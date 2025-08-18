import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from typing import Optional

import pandas as pd
import numpy as np


class MetricsError(Exception):
    pass


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the DataFrame for metrics calculation."""
    if df.empty:
        raise MetricsError("Input CSV has no rows.")
    if "rating" not in df.columns:
        raise MetricsError("Column 'rating' is required.")
    
    # Keep only valid ratings 1..5
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1, 5, inclusive="both")]
    if df.empty:
        raise MetricsError("No valid ratings in input.")
    return df


def _rating_distribution(df: pd.DataFrame):
    """Calculate rating distribution and percentages."""
    counts = df["rating"].value_counts().reindex([1,2,3,4,5], fill_value=0).astype(int)
    total = int(counts.sum())
    pct = (counts / max(total, 1) * 100).round(2)
    
    return {
        "counts": {int(k): int(v) for k, v in counts.to_dict().items()},
        "percentages": {int(k): float(v) for k, v in pct.to_dict().items()},
        "total": total,
        "average_rating": round(float(df["rating"].mean()), 2),
        "median_rating": float(df["rating"].median())
    }


def _text_metrics(df: pd.DataFrame):
    """Calculate text-related metrics if text data is available."""
    metrics = {}
    
    # Use processed text lengths if available
    if 'text_length' in df.columns:
        lengths = df['text_length'].dropna()
        if not lengths.empty:
            metrics["text_stats"] = {
                "avg_length": round(float(lengths.mean()), 1),
                "median_length": float(lengths.median()),
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "std_length": round(float(lengths.std()), 1)
            }
    
    # Use word counts if available
    if 'word_count' in df.columns:
        word_counts = df['word_count'].dropna()
        if not word_counts.empty:
            metrics["word_stats"] = {
                "avg_words": round(float(word_counts.mean()), 1),
                "median_words": float(word_counts.median()),
                "min_words": int(word_counts.min()),
                "max_words": int(word_counts.max()),
                "std_words": round(float(word_counts.std()), 1)
            }
    
    # Content presence flags
    if 'has_title' in df.columns and 'has_review' in df.columns:
        metrics["content_presence"] = {
            "has_title": int(df['has_title'].sum()),
            "has_review": int(df['has_review'].sum()),
            "title_only": int((df['has_title'] & ~df['has_review']).sum()),
            "review_only": int((~df['has_title'] & df['has_review']).sum()),
            "both_title_and_review": int((df['has_title'] & df['has_review']).sum())
        }
    
    return metrics


def _summarize(df: pd.DataFrame):
    """Calculate comprehensive metrics summary."""
    df = _validate(df)
    
    # Core rating metrics
    rating_metrics = _rating_distribution(df)
    
    # Text metrics (if available)
    text_metrics = _text_metrics(df)
    
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_reviews": rating_metrics["total"],
        "rating_metrics": rating_metrics,
    }
    
    # Add text metrics if any were calculated
    if text_metrics:
        summary["text_metrics"] = text_metrics
    
    # Breakdowns by country
    if "country" in df.columns:
        by_country = {}
        country_counts = df['country'].value_counts()
        for c, sub in df.groupby("country", dropna=False):
            if len(sub) < 1:  # Skip empty groups
                continue
            try:
                sub_validated = _validate(sub)
                by_country[str(c)] = {
                    "review_count": len(sub_validated),
                    "average_rating": round(float(sub_validated["rating"].mean()), 2),
                    "rating_distribution": _rating_distribution(sub_validated)
                }
            except MetricsError:
                # Skip countries with no valid ratings
                continue
        if by_country:
            summary["by_country"] = by_country
    
    # Breakdowns by version (top versions only)
    if "version" in df.columns:
        version_counts = df['version'].value_counts()
        top_versions = version_counts.head(10)  # Only top 10 versions
        by_version = {}
        
        for v, sub in df.groupby("version", dropna=False):
            if str(v) not in top_versions.index or len(sub) < 5:  # Skip rare versions
                continue
            try:
                sub_validated = _validate(sub)
                by_version[str(v)] = {
                    "review_count": len(sub_validated),
                    "average_rating": round(float(sub_validated["rating"].mean()), 2),
                    "rating_distribution": _rating_distribution(sub_validated)
                }
            except MetricsError:
                continue
        if by_version:
            summary["by_version"] = by_version
    
    # Temporal analysis if date available
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            summary["temporal_metrics"] = {
                "date_range": {
                    "earliest": dates.min().strftime('%Y-%m-%d'),
                    "latest": dates.max().strftime('%Y-%m-%d'),
                    "span_days": (dates.max() - dates.min()).days
                },
                "reviews_by_month": {str(k): int(v) for k, v in dates.dt.to_period('M').value_counts().sort_index().tail(12).to_dict().items()}
            }
    
    return summary


def _write_outputs(summary: dict, input_csv: Path, out_json: Path, out_csv_summary: Optional[Path]):
    """Write metrics outputs to JSON and optional CSV summary."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Generate human-readable summary report
    report_path = out_json.with_suffix('.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(_generate_summary_report(summary))

    # Optional flat CSV summary (handy for quick checks)
    if out_csv_summary:
        rows = []
        rating_metrics = summary.get("rating_metrics", {})
        
        # Overall row
        rows.append({
            "scope": "overall",
            "key": "",
            "total_reviews": rating_metrics.get("total", 0),
            "average_rating": rating_metrics.get("average_rating", 0),
            "pct_1": rating_metrics.get("percentages", {}).get(1, 0),
            "pct_2": rating_metrics.get("percentages", {}).get(2, 0),
            "pct_3": rating_metrics.get("percentages", {}).get(3, 0),
            "pct_4": rating_metrics.get("percentages", {}).get(4, 0),
            "pct_5": rating_metrics.get("percentages", {}).get(5, 0),
        })
        
        # By country
        if "by_country" in summary:
            for k, v in summary["by_country"].items():
                rows.append({
                    "scope": "country",
                    "key": k,
                    "total_reviews": v["rating_distribution"]["total"],
                    "average_rating": v["average_rating"],
                    "pct_1": v["rating_distribution"]["percentages"][1],
                    "pct_2": v["rating_distribution"]["percentages"][2],
                    "pct_3": v["rating_distribution"]["percentages"][3],
                    "pct_4": v["rating_distribution"]["percentages"][4],
                    "pct_5": v["rating_distribution"]["percentages"][5],
                })
        
        # By version
        if "by_version" in summary:
            for k, v in summary["by_version"].items():
                rows.append({
                    "scope": "version",
                    "key": k,
                    "total_reviews": v["rating_distribution"]["total"],
                    "average_rating": v["average_rating"],
                    "pct_1": v["rating_distribution"]["percentages"][1],
                    "pct_2": v["rating_distribution"]["percentages"][2],
                    "pct_3": v["rating_distribution"]["percentages"][3],
                    "pct_4": v["rating_distribution"]["percentages"][4],
                    "pct_5": v["rating_distribution"]["percentages"][5],
                })
        
        pd.DataFrame(rows).to_csv(out_csv_summary, index=False)

    print(f"✓ Metrics JSON: {out_json}")
    print(f"✓ Summary report: {report_path}")
    if out_csv_summary:
        print(f"✓ Metrics CSV: {out_csv_summary}")


def _generate_summary_report(summary: dict) -> str:
    """Generate a human-readable summary report."""
    lines = [
        "=== APP REVIEW METRICS SUMMARY ===",
        f"Generated: {summary.get('generated_at', 'Unknown')}",
        f"Total Reviews: {summary.get('total_reviews', 0):,}",
        ""
    ]
    
    # Rating summary
    if "rating_metrics" in summary:
        rating = summary["rating_metrics"]
        lines.extend([
            f"Average Rating: {rating.get('average_rating', 0):.2f}/5.0",
            f"Median Rating: {rating.get('median_rating', 0):.1f}/5.0",
            "",
            "Rating Distribution:",
            f"  ⭐⭐⭐⭐⭐ 5 stars: {rating.get('counts', {}).get(5, 0):,} ({rating.get('percentages', {}).get(5, 0):.1f}%)",
            f"  ⭐⭐⭐⭐   4 stars: {rating.get('counts', {}).get(4, 0):,} ({rating.get('percentages', {}).get(4, 0):.1f}%)",
            f"  ⭐⭐⭐     3 stars: {rating.get('counts', {}).get(3, 0):,} ({rating.get('percentages', {}).get(3, 0):.1f}%)",
            f"  ⭐⭐       2 stars: {rating.get('counts', {}).get(2, 0):,} ({rating.get('percentages', {}).get(2, 0):.1f}%)",
            f"  ⭐         1 star:  {rating.get('counts', {}).get(1, 0):,} ({rating.get('percentages', {}).get(1, 0):.1f}%)",
        ])
    
    # Text metrics
    if "text_metrics" in summary:
        text = summary["text_metrics"]
        lines.append("")
        lines.append("Text Analysis:")
        
        if "text_stats" in text:
            stats = text["text_stats"]
            lines.append(f"  Average Length: {stats.get('avg_length', 0):.0f} characters")
            lines.append(f"  Length Range: {stats.get('min_length', 0)}-{stats.get('max_length', 0)} characters")
        
        if "word_stats" in text:
            words = text["word_stats"]
            lines.append(f"  Average Words: {words.get('avg_words', 0):.0f} words")
            lines.append(f"  Word Range: {words.get('min_words', 0)}-{words.get('max_words', 0)} words")
    
    # Country breakdown
    if "by_country" in summary:
        lines.append("")
        lines.append("By Country (Top 10):")
        countries = summary["by_country"]
        sorted_countries = sorted(countries.items(), 
                                key=lambda x: x[1]["review_count"], reverse=True)
        for country, data in sorted_countries[:10]:
            lines.append(f"  {country.upper()}: {data['review_count']:,} reviews, "
                        f"{data['average_rating']:.1f}★ avg")
    
    return "\n".join(lines)


def main():
    """Main entry point for metrics calculation."""
    ap = argparse.ArgumentParser(description="Calculate metrics from app review CSV data.")
    ap.add_argument("input", help="Path to CSV file (scraped or processed)")
    ap.add_argument("--output", "-o", help="Output JSON path (default: <input>.metrics.json)")
    ap.add_argument("--csv-summary", help="Optional: also write flat CSV summary")
    ap.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    
    args = ap.parse_args()

    try:
        input_csv = Path(args.input)
        if not input_csv.exists():
            raise MetricsError(f"Input not found: {input_csv}")

        if not args.quiet:
            print(f"📊 Calculating metrics for {input_csv}")
        
        df = pd.read_csv(input_csv)
        summary = _summarize(df)

        out_json = Path(args.output) if args.output else input_csv.with_suffix(".metrics.json")
        out_csv = Path(args.csv_summary) if args.csv_summary else None
        
        _write_outputs(summary, input_csv, out_json, out_csv)
        
        # Print summary to console unless quiet
        if not args.quiet:
            print("\n" + _generate_summary_report(summary))

    except MetricsError as e:
        print(f"METRICS ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
