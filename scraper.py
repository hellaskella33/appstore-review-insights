import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"

# Major App Store countries for comprehensive review fetching
DEFAULT_COUNTRIES = [
    "us", "gb", "ca", "au", "de", "fr", "it", "es", "nl", "br", 
    "jp", "kr", "cn", "in", "mx", "ar", "se", "no", "dk", "fi"
]


class ScraperError(Exception):
    pass


class ValidationError(Exception):
    pass


def validate_country_codes(countries: List[str]) -> List[str]:
    """
    Validate country codes and return normalized list.
    """
    if not countries:
        raise ValidationError("At least one country code is required")
    
    valid_countries = []
    invalid_countries = []
    
    for country in countries:
        country = country.strip().lower()
        if not country:
            continue
        if len(country) != 2 or not country.isalpha():
            invalid_countries.append(country)
            continue
        valid_countries.append(country)
    
    if invalid_countries:
        raise ValidationError(f"Invalid country codes: {', '.join(invalid_countries)}. Use 2-letter codes like 'us', 'gb', 'de'")
    
    if not valid_countries:
        raise ValidationError("No valid country codes provided")
    
    return valid_countries


def validate_app_inputs(app_id: Optional[int], app_name: Optional[str]) -> None:
    """
    Validate app identification inputs.
    """
    if not app_id and not app_name:
        raise ValidationError("Either --app-id or --app-name must be provided")
    
    if app_id is not None:
        if app_id <= 0:
            raise ValidationError(f"app_id must be a positive integer, got: {app_id}")
        if app_id > 999999999999:  # Reasonable upper bound for iTunes IDs
            raise ValidationError(f"app_id seems too large, got: {app_id}")
    
    if app_name is not None:
        app_name = app_name.strip()
        if not app_name:
            raise ValidationError("app_name cannot be empty")
        if len(app_name) > 200:  # Reasonable upper bound
            raise ValidationError(f"app_name too long ({len(app_name)} chars), max 200 characters")


def validate_numeric_params(how_many: int, sample_size: int, seed: int) -> None:
    """
    Validate numeric parameters.
    """
    if how_many <= 0:
        raise ValidationError(f"how_many must be positive, got: {how_many}")
    if how_many > 5000:  # Reasonable upper bound to prevent excessive API calls
        raise ValidationError(f"how_many too large ({how_many}), max 5000 per country")
    
    if sample_size <= 0:
        raise ValidationError(f"sample_size must be positive, got: {sample_size}")
    if sample_size > 50000:  # Reasonable upper bound
        raise ValidationError(f"sample_size too large ({sample_size}), max 50000")
    
    if seed < 0:
        raise ValidationError(f"seed must be non-negative, got: {seed}")


def resolve_app_id_by_name(app_name: str, country: str, limit: int = 5) -> Tuple[int, str]:
    """
    Resolve a numeric app_id (trackId) from an app_name using Apple's public Search API.
    Returns (app_id, resolved_app_name). Raises if not found.
    """
    if not app_name or not app_name.strip():
        raise ValidationError("app_name cannot be empty")
    if not country or len(country) != 2:
        raise ValidationError(f"Invalid country code: '{country}'")
    
    params = {
        "term": app_name.strip(),
        "entity": "software",
        "country": country.lower(),
        "limit": max(1, min(limit, 50)),  # Clamp limit to reasonable range
    }
    
    try:
        r = requests.get(ITUNES_SEARCH_URL, params=params, timeout=30)
        r.raise_for_status()  # Raises HTTPError for bad responses
    except requests.exceptions.Timeout:
        raise ScraperError(f"iTunes search timed out for app '{app_name}' in {country}")
    except requests.exceptions.ConnectionError:
        raise ScraperError(f"Failed to connect to iTunes API. Check your internet connection.")
    except requests.exceptions.HTTPError:
        raise ScraperError(f"iTunes search failed: HTTP {r.status_code} - {r.reason}")
    except requests.exceptions.RequestException as e:
        raise ScraperError(f"iTunes search request failed: {e}")
    
    try:
        data = r.json()
    except json.JSONDecodeError:
        raise ScraperError("iTunes API returned invalid JSON response")
    
    if not isinstance(data, dict):
        raise ScraperError("iTunes API returned unexpected response format")
    
    results = data.get("results", [])
    if not results:
        raise ScraperError(f"No apps found for '{app_name}' in {country.upper()} App Store. Try different spelling or country.")

    # Choose the best exact/contains match first
    app_name_lower = app_name.strip().lower()
    exact_matches = [x for x in results if x.get("trackName", "").strip().lower() == app_name_lower]
    candidate = exact_matches[0] if exact_matches else results[0]
    
    app_id = candidate.get("trackId")
    resolved_name = candidate.get("trackName", "").strip()
    
    if not app_id:
        raise ScraperError("Found app but missing trackId in response")
    if not isinstance(app_id, int) or app_id <= 0:
        raise ScraperError(f"Invalid trackId received: {app_id}")
    if not resolved_name:
        resolved_name = app_name  # Fallback to original name
    
    return app_id, resolved_name


def search_apps_by_name(app_name: str, country: str = "us", limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for apps by name and return multiple results for user selection.
    Returns list of app dictionaries with id, name, developer, etc.
    """
    if not app_name or not app_name.strip():
        raise ValidationError("app_name cannot be empty")
    if not country or len(country) != 2:
        raise ValidationError(f"Invalid country code: '{country}'")
    
    params = {
        "term": app_name.strip(),
        "entity": "software", 
        "country": country.lower(),
        "limit": max(1, min(limit, 50)),
    }
    
    try:
        r = requests.get(ITUNES_SEARCH_URL, params=params, timeout=30)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise ScraperError(f"iTunes search timed out for app '{app_name}' in {country}")
    except requests.exceptions.ConnectionError:
        raise ScraperError(f"Failed to connect to iTunes API. Check your internet connection.")
    except requests.exceptions.HTTPError:
        raise ScraperError(f"iTunes search failed: HTTP {r.status_code} - {r.reason}")
    except requests.exceptions.RequestException as e:
        raise ScraperError(f"iTunes search request failed: {e}")
    
    try:
        data = r.json()
    except json.JSONDecodeError:
        raise ScraperError("Invalid JSON response from iTunes API")
    
    if 'results' not in data:
        raise ScraperError("No 'results' key in iTunes API response")
    
    results = []
    for item in data['results']:
        if item.get('wrapperType') == 'software' and 'trackId' in item:
            results.append({
                'id': item['trackId'],
                'name': item.get('trackName', 'Unknown'),
                'developer': item.get('artistName', 'Unknown'),
                'category': item.get('primaryGenreName', 'Unknown'),
                'price': item.get('formattedPrice', 'Unknown'),
                'rating': item.get('averageUserRating', 0.0),
                'icon_url': item.get('artworkUrl100', ''),
                'description': item.get('description', '')[:200] + ('...' if len(item.get('description', '')) > 200 else ''),
                'bundle_id': item.get('bundleId', ''),
                'version': item.get('version', ''),
                'release_date': item.get('releaseDate', ''),
                'file_size': item.get('fileSizeBytes', 0)
            })
    
    return results


def verify_app_exists(app_id: int, country: str) -> Dict[str, Any]:
    """
    Verify app exists via iTunes Lookup and return basic metadata dict.
    """
    if not isinstance(app_id, int) or app_id <= 0:
        raise ValidationError(f"Invalid app_id: {app_id}")
    if not country or len(country) != 2:
        raise ValidationError(f"Invalid country code: '{country}'")
    
    params = {"id": app_id, "country": country.lower()}
    
    try:
        r = requests.get(ITUNES_LOOKUP_URL, params=params, timeout=30)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise ScraperError(f"iTunes lookup timed out for app_id {app_id} in {country}")
    except requests.exceptions.ConnectionError:
        raise ScraperError(f"Failed to connect to iTunes API. Check your internet connection.")
    except requests.exceptions.HTTPError:
        raise ScraperError(f"iTunes lookup failed: HTTP {r.status_code} - {r.reason}")
    except requests.exceptions.RequestException as e:
        raise ScraperError(f"iTunes lookup request failed: {e}")
    
    try:
        data = r.json()
    except json.JSONDecodeError:
        raise ScraperError("iTunes API returned invalid JSON response")
    
    if not isinstance(data, dict):
        raise ScraperError("iTunes API returned unexpected response format")
    
    results = data.get("results", [])
    if not results:
        raise ScraperError(f"App with ID {app_id} not found in {country.upper()} App Store. Check app_id and country.")
    
    app_info = results[0]
    if not isinstance(app_info, dict):
        raise ScraperError("Invalid app data format in iTunes response")
    
    return app_info


def parse_review_entry(entry) -> Optional[Dict[str, Any]]:
    """
    Parse a single RSS review entry with robust error handling and fallbacks.
    Returns None if parsing fails completely.
    """
    try:
        import xml.etree.ElementTree as ET
        
        # Extract review data with fallbacks and validation
        def safe_extract_text(element, fallback=""):
            """Safely extract text from XML element with fallback."""
            if element is not None and hasattr(element, 'text'):
                text = element.text
                if text and isinstance(text, str):
                    return text.strip()
            return fallback
        
        def safe_extract_int(element, fallback=None):
            """Safely extract integer from XML element with fallback."""
            if element is not None and hasattr(element, 'text'):
                try:
                    return int(element.text.strip())
                except (ValueError, TypeError, AttributeError):
                    pass
            return fallback
        
        # Find elements with namespace handling
        title_elem = entry.find(".//{http://www.w3.org/2005/Atom}title")
        content_elem = entry.find(".//{http://www.w3.org/2005/Atom}content")
        author_elem = entry.find(".//{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name")
        updated_elem = entry.find(".//{http://www.w3.org/2005/Atom}updated")
        rating_elem = entry.find(".//{http://itunes.apple.com/rss}rating")
        version_elem = entry.find(".//{http://itunes.apple.com/rss}version")
        
        # Build review dict with safe extraction
        review = {
            "userName": safe_extract_text(author_elem, "Anonymous"),
            "title": safe_extract_text(title_elem, ""),
            "review": safe_extract_text(content_elem, ""),
            "rating": safe_extract_int(rating_elem, None),
            "date": safe_extract_text(updated_elem, ""),
            "version": safe_extract_text(version_elem, ""),
            "isEdited": False,  # Not available in RSS
            "voteCount": 0,     # Not available in RSS
            "developerResponse": None  # Not available in RSS
        }
        
        # Basic validation - skip entries that are completely empty
        if (not review["userName"] or review["userName"] == "Anonymous") and \
           not review["title"] and not review["review"]:
            return None
            
        # Validate rating if present
        if review["rating"] is not None:
            if not isinstance(review["rating"], int) or review["rating"] < 1 or review["rating"] > 5:
                review["rating"] = None  # Invalid rating, set to None
        
        # Clean up text fields
        for field in ["userName", "title", "review", "version"]:
            if review[field] and len(review[field]) > 10000:  # Reasonable max length
                review[field] = review[field][:10000] + "..."
        
        return review
        
    except Exception as e:
        # Return None for completely failed parsing
        return None


def fetch_reviews_multi_country(
    app_id: Optional[int],
    app_name: Optional[str],
    countries: List[str],
    how_many_per_country: int,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Fetch reviews from multiple countries and combine them.
    """
    all_reviews = []
    
    for country in countries:
        try:
            print(f"Fetching reviews from {country.upper()}...")
            reviews = fetch_reviews_with_retries(
                app_id=app_id,
                app_name=app_name,
                country=country,
                how_many=how_many_per_country,
                max_retries=max_retries
            )
            
            # Add country info to each review
            for review in reviews:
                review["country"] = country
            
            all_reviews.extend(reviews)
            print(f"Fetched {len(reviews)} reviews from {country.upper()}")
            
        except ScraperError as e:
            print(f"Warning: Failed to fetch from {country.upper()}: {e}")
            continue
        except Exception as e:
            print(f"Warning: Unexpected error for {country.upper()}: {e}")
            continue
    
    return all_reviews


def fetch_reviews_with_retries(
    app_id: Optional[int],
    app_name: Optional[str],
    country: str,
    how_many: int,
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Fetch reviews using Apple's RSS feed API directly.
    Retries on transient failures.
    """
    if not app_id and not app_name:
        raise ValueError("Either app_id or app_name must be provided.")

    # If we don't have app_id, resolve it first
    if not app_id:
        app_id, _ = resolve_app_id_by_name(app_name, country)

    attempt = 0
    last_err: Optional[Exception] = None
    while attempt <= max_retries:
        try:
            # Validate inputs
            if not isinstance(app_id, int) or app_id <= 0:
                raise ValidationError(f"Invalid app_id: {app_id}")
            if not country or len(country) != 2:
                raise ValidationError(f"Invalid country code: '{country}'")
            
            # Use Apple's RSS feed for reviews
            rss_url = f"https://itunes.apple.com/{country.lower()}/rss/customerreviews/id={app_id}/sortBy=mostRecent/xml"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            try:
                response = requests.get(rss_url, headers=headers, timeout=30)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                raise ScraperError(f"RSS feed request timed out for app_id {app_id} in {country}")
            except requests.exceptions.ConnectionError:
                raise ScraperError(f"Failed to connect to Apple RSS feed. Check your internet connection.")
            except requests.exceptions.HTTPError:
                if response.status_code == 404:
                    raise ScraperError(f"No reviews RSS feed found for app_id {app_id} in {country.upper()}. App may not exist in this country.")
                elif response.status_code == 403:
                    raise ScraperError(f"Access denied to RSS feed for app_id {app_id} in {country.upper()}. App may be restricted.")
                else:
                    raise ScraperError(f"RSS feed request failed: HTTP {response.status_code} - {response.reason}")
            except requests.exceptions.RequestException as e:
                raise ScraperError(f"RSS feed request failed: {e}")
            
            if not response.content:
                raise ScraperError(f"Empty response from RSS feed for app_id {app_id} in {country}")
            
            # Parse XML response with robust error handling
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                raise ScraperError(f"Invalid XML in RSS feed response: {e}")
            
            # Find all review entries
            reviews = []
            try:
                entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            except Exception as e:
                raise ScraperError(f"Failed to parse RSS feed structure: {e}")
            
            if not entries:
                # This is common - apps may have no reviews in some countries
                return []
            
            processed_count = 0
            for entry in entries:
                if processed_count >= how_many:
                    break
                    
                try:
                    # Extract review data with fallbacks
                    review = parse_review_entry(entry)
                    if review:  # Only add if parsing succeeded
                        reviews.append(review)
                        processed_count += 1
                except Exception as e:
                    # Log but continue with other reviews
                    print(f"Warning: Failed to parse review entry: {e}")
                    continue
            
            return reviews
            
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            sleep_s = (backoff_base ** attempt) + random.random()
            time.sleep(sleep_s)
            attempt += 1
    raise ScraperError(f"Failed to fetch reviews after {max_retries+1} attempts: {last_err}")


def dedupe_reviews(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Basic in-run deduplication by a composite key of (userName|title|review|date).
    """
    seen = set()
    out = []
    for r in rows:
        key = (
            str(r.get("userName", "")).strip(),
            str(r.get("title", "")).strip(),
            str(r.get("review", "")).strip(),
            str(r.get("date", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def random_sample(rows: List[Dict[str, Any]], sample_size: Optional[int], seed: int) -> List[Dict[str, Any]]:
    """
    Deterministic random sampling. If sample_size is None, return rows unchanged.
    If requested sample > available, return all rows.
    """
    if sample_size is None:
        return rows
    n = min(sample_size, len(rows))
    rnd = random.Random(seed)
    idxs = list(range(len(rows)))
    rnd.shuffle(idxs)
    chosen = idxs[:n]
    return [rows[i] for i in chosen]


def normalize_dataframe(raw_reviews: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw list of dicts to a normalized DataFrame with a stable schema.
    No cleaning/sentiment — scrape-only.
    """
    df = pd.DataFrame(raw_reviews)

    # Ensure stable columns (fill missing with None)
    cols = [
        "userName",
        "title",
        "rating",
        "date",
        "version",
        "isEdited",
        "voteCount",
        "review",
        "developerResponse",
        "country",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # date can be string or datetime — keep as ISO8601 string for CSV stability
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Stable order
    df = df[
        [
            "userName",
            "title",
            "rating",
            "date",
            "version",
            "isEdited",
            "voteCount",
            "review",
            "developerResponse",
            "country",
        ]
    ]
    return df


def write_outputs(
    df: pd.DataFrame,
    out_dir: Path,
    base_name: str,
    meta: Dict[str, Any],
) -> Tuple[Path, Path]:
    """
    Write CSV and sidecar meta JSON. Returns paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    csv_path = out_dir / f"{base_name}_{ts}.csv"
    json_path = out_dir / f"{base_name}_{ts}.meta.json"

    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Scrape Apple App Store reviews → CSV (scraper-only).")
    parser.add_argument("--country", help="Single country code, e.g. us, gb, de. If not specified, fetches from all major countries.")
    parser.add_argument("--countries", help="Comma-separated list of country codes, e.g. 'us,gb,de,fr'.")
    parser.add_argument("--app-id", type=int, help="Numeric trackId of the app (preferred for stability).")
    parser.add_argument("--app-name", help="If no app-id, resolve by name via iTunes Search API.")
    parser.add_argument("--how-many", type=int, default=60, help="How many recent reviews to fetch per country.")
    parser.add_argument("--sample-size", type=int, default=100, help="Random sample size for the final output.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--out-dir", default="data", help="Output directory.")
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="If set, writes all fetched reviews without random sampling.",
    )

    args = parser.parse_args()

    try:
        # Validate inputs early
        validate_app_inputs(args.app_id, args.app_name)
        validate_numeric_params(args.how_many, args.sample_size, args.seed)
        
        # Determine and validate countries to fetch from
        if args.countries:
            countries = validate_country_codes([c.strip() for c in args.countries.split(",")])
        elif args.country:
            countries = validate_country_codes([args.country])
        else:
            countries = DEFAULT_COUNTRIES
            print(f"No country specified, fetching from {len(countries)} major countries: {', '.join(countries)}")

        # Resolve app_id if needed (use first country for resolution)
        resolved_name = args.app_name
        app_id = args.app_id
        if not app_id and not args.app_name:
            raise ScraperError("Provide --app-id OR --app-name.")

        primary_country = countries[0]
        if not app_id and args.app_name:
            app_id, resolved_name = resolve_app_id_by_name(args.app_name, primary_country)

        # Verify the app exists (helps catch wrong storefront)
        app_meta = verify_app_exists(app_id, primary_country)
        bundle_id = app_meta.get("bundleId")
        official_name = app_meta.get("trackName", resolved_name) or resolved_name

        # Fetch from multiple countries
        if len(countries) == 1:
            raw = fetch_reviews_with_retries(
                app_id=app_id,
                app_name=None,
                country=countries[0],
                how_many=args.how_many,
            )
            # Add country info for single-country scrapes too
            for review in raw:
                review["country"] = countries[0]
        else:
            raw = fetch_reviews_multi_country(
                app_id=app_id,
                app_name=None,
                countries=countries,
                how_many_per_country=args.how_many,
            )
        if not raw:
            raise ScraperError("Fetched zero reviews.")

        # In-run dedupe
        raw = dedupe_reviews(raw)

        # Optional sampling (scrape-only, no cleaning)
        sampled = raw if args.no_sample else random_sample(raw, args.sample_size, args.seed)

        # Normalize to a stable CSV schema
        df = normalize_dataframe(sampled)

        # Write outputs
        country_suffix = "global" if len(countries) > 1 else countries[0]
        base_name = f"{official_name.replace(' ', '_').lower()}_{country_suffix}_reviews"
        csv_path, meta_path = write_outputs(
            df=df,
            out_dir=Path(args.out_dir),
            base_name=base_name,
            meta={
                "schema_version": "1.0.0",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source": "apple-rss-feed",
                "countries": countries,
                "app_id": app_id,
                "app_name": official_name,
                "bundle_id": bundle_id,
                "requested_how_many_per_country": args.how_many,
                "written_rows": int(df.shape[0]),
                "random_sample": not args.no_sample,
                "seed": args.seed if not args.no_sample else None,
            },
        )

        print(f"OK: wrote {csv_path} and {meta_path}")
    except ValidationError as e:
        print(f"INPUT ERROR: {e}", file=sys.stderr)
        print("Use --help for usage information.", file=sys.stderr)
        sys.exit(1)
    except ScraperError as e:
        print(f"SCRAPER ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except MemoryError:
        print("ERROR: Out of memory. Try reducing --how-many or --sample-size parameters.", file=sys.stderr)
        sys.exit(4)
    except OSError as e:
        print(f"FILE ERROR: {e}", file=sys.stderr)
        print("Check output directory permissions and disk space.", file=sys.stderr)
        sys.exit(5)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        print("This may be a bug. Please report it with the full error message.", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
