import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import argparse

import pandas as pd
from openai import OpenAI

from composer import ReviewComposer, ComposerError


class InsightsError(Exception):
    pass


class ReviewInsightsGenerator:
    """
    LLM-powered insights generation from composed review analysis.
    Generates business insights, recommendations, and strategic analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the insights generator.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use for analysis
        """
        self.model = model
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try to get from environment
            api_key_env = os.getenv('OPENAI_API_KEY')
            if not api_key_env:
                raise InsightsError("OpenAI API key required. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
            self.client = OpenAI(api_key=api_key_env)
        
        print(f"Initialized InsightsGenerator with model: {model}")
    
    def generate_comprehensive_insights(self, composed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive business insights from composed analysis data.
        
        Args:
            composed_data: Results from ReviewComposer
            
        Returns:
            Dictionary with comprehensive insights
        """
        print("🧠 Generating comprehensive insights with LLM...")
        
        # Extract key information
        composition_info = composed_data.get("composition_info", {})
        insights_prep = composed_data.get("insights_preparation", {})
        summary = composed_data.get("summary", {})
        
        # Generate different types of insights
        results = {
            "generation_info": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "llm_model": self.model,
                "source_data": {
                    "total_reviews": composition_info.get("total_reviews", 0),
                    "negative_reviews": composition_info.get("negative_reviews", 0),
                    "topics_discovered": composition_info.get("topics_discovered", 0)
                }
            }
        }
        
        # Generate executive summary
        print("📊 Generating executive summary...")
        results["executive_summary"] = self._generate_executive_summary(insights_prep, summary)
        
        # Generate topic-specific insights
        print("🎯 Generating topic-specific insights...")
        results["topic_insights"] = self._generate_topic_insights(insights_prep)
        
        # Generate strategic recommendations
        print("💡 Generating strategic recommendations...")
        results["strategic_recommendations"] = self._generate_recommendations(insights_prep, summary)
        
        # Generate priority matrix
        print("⚡ Generating priority matrix...")
        results["priority_matrix"] = self._generate_priority_matrix(insights_prep, summary)
        
        # Generate risk assessment
        print("⚠️ Generating risk assessment...")
        results["risk_assessment"] = self._generate_risk_assessment(insights_prep)
        
        return results
    
    def _generate_executive_summary(self, insights_prep: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary using LLM."""
        main_context = insights_prep.get("prompts_for_llm", {}).get("main_context", "")
        key_findings = summary.get("key_findings", {})
        
        prompt = f"""As a senior product analyst, provide an executive summary of this app review analysis.

{main_context}

Focus on:
1. Overall sentiment and user satisfaction
2. Main complaint categories and their business impact
3. Key takeaways for leadership
4. Critical areas requiring immediate attention

Provide a concise but comprehensive executive summary in 3-4 paragraphs.
Be specific about percentages, numbers, and actionable insights."""

        try:
            response = self._call_openai(prompt)
            
            # Parse key metrics from the data
            overview = insights_prep.get("overview", {})
            negative_analysis = insights_prep.get("negative_review_analysis", {})
            
            return {
                "summary_text": response,
                "key_metrics": {
                    "total_reviews_analyzed": overview.get("review_distribution", {}).get("total_reviews", 0),
                    "average_rating": overview.get("review_distribution", {}).get("average_rating", 0),
                    "negative_review_percentage": self._calculate_negative_percentage(overview),
                    "main_complaint_categories": len(negative_analysis.get("topic_breakdown", [])),
                    "top_complaint": self._get_top_complaint(negative_analysis)
                }
            }
        
        except Exception as e:
            return {
                "summary_text": f"Error generating executive summary: {e}",
                "key_metrics": {}
            }
    
    def _generate_topic_insights(self, insights_prep: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed insights for each topic."""
        topic_contexts = insights_prep.get("prompts_for_llm", {}).get("topic_contexts", {})
        negative_analysis = insights_prep.get("negative_review_analysis", {})
        topic_insights = {}
        
        for topic_breakdown in negative_analysis.get("topic_breakdown", []):
            topic_id = topic_breakdown["topic_id"]
            topic_context = topic_contexts.get(topic_id, "")
            
            prompt = f"""As a product analyst, analyze this specific complaint theme:

{topic_context}

Provide detailed analysis covering:
1. **Root Cause Analysis**: What underlying issues are causing these complaints?
2. **Business Impact**: How might this theme affect user retention, ratings, and growth?
3. **User Experience Impact**: What specific UX/functionality problems are indicated?
4. **Severity Assessment**: Rate the severity (High/Medium/Low) and explain why
5. **Quick Wins**: What could be fixed quickly to address some complaints?
6. **Long-term Solutions**: What larger changes might be needed?

Be specific and actionable in your analysis."""

            try:
                analysis = self._call_openai(prompt)
                
                topic_insights[str(topic_id)] = {
                    "topic_name": topic_breakdown["topic_name"],
                    "review_count": topic_breakdown["review_count"],
                    "percentage_of_negative": topic_breakdown["percentage"],
                    "detailed_analysis": analysis,
                    "key_phrases": topic_breakdown["key_phrases"],
                    "sample_reviews": topic_breakdown["sample_reviews"]
                }
                
            except Exception as e:
                topic_insights[str(topic_id)] = {
                    "topic_name": topic_breakdown["topic_name"],
                    "detailed_analysis": f"Error generating analysis: {e}",
                    "review_count": topic_breakdown["review_count"],
                    "percentage_of_negative": topic_breakdown["percentage"]
                }
        
        return topic_insights
    
    def _generate_recommendations(self, insights_prep: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic recommendations."""
        main_context = insights_prep.get("prompts_for_llm", {}).get("main_context", "")
        key_findings = summary.get("key_findings", {})
        
        prompt = f"""As a senior product strategist, provide comprehensive recommendations based on this analysis:

{main_context}

Provide recommendations in these categories:

1. **Immediate Actions (0-30 days)**
   - Critical fixes needed right away
   - Quick wins to improve sentiment

2. **Short-term Improvements (1-3 months)**
   - Feature enhancements
   - UX improvements
   - Process changes

3. **Long-term Strategy (3-12 months)**
   - Major feature development
   - Platform improvements
   - Strategic initiatives

4. **Resource Allocation**
   - What teams/skills are needed
   - Priority order for addressing issues

For each recommendation, include:
- Specific action items
- Expected impact on user satisfaction
- Estimated effort/complexity
- Success metrics to track

Be concrete and actionable."""

        try:
            recommendations_text = self._call_openai(prompt)
            
            return {
                "recommendations_text": recommendations_text,
                "priority_areas": self._extract_priority_areas(insights_prep),
                "estimated_impact": self._estimate_impact(insights_prep)
            }
            
        except Exception as e:
            return {
                "recommendations_text": f"Error generating recommendations: {e}",
                "priority_areas": [],
                "estimated_impact": {}
            }
    
    def _generate_priority_matrix(self, insights_prep: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate priority matrix for complaint themes."""
        negative_analysis = insights_prep.get("negative_review_analysis", {})
        
        prompt = f"""As a product manager, create a priority matrix for these complaint themes:

"""
        
        # Add each topic to the prompt
        for topic in negative_analysis.get("topic_breakdown", []):
            prompt += f"""
Theme: {topic['topic_name']}
- Affects {topic['review_count']} reviews ({topic['percentage']}% of negative feedback)
- Key issues: {', '.join(topic['key_phrases'])}
- Sample complaint: "{topic['sample_reviews'][0] if topic['sample_reviews'] else 'N/A'}"
"""
        
        prompt += """
For each theme, assess:
1. **Impact Score (1-10)**: How much this affects user experience and business metrics
2. **Effort Score (1-10)**: How difficult/expensive it would be to address (1=easy, 10=very hard)
3. **Priority Category**: High/Medium/Low based on impact vs effort
4. **Recommended Timeline**: When to address this (Immediate/Short-term/Long-term)

Format your response as analysis text explaining the prioritization logic."""

        try:
            priority_analysis = self._call_openai(prompt)
            
            # Create structured priority data
            priority_matrix = {
                "analysis": priority_analysis,
                "themes_by_priority": self._categorize_by_priority(negative_analysis),
                "methodology": "Prioritization based on impact (user satisfaction, business metrics) vs effort (development complexity, resources needed)"
            }
            
            return priority_matrix
            
        except Exception as e:
            return {
                "analysis": f"Error generating priority matrix: {e}",
                "themes_by_priority": {"high": [], "medium": [], "low": []},
                "methodology": "Error in analysis"
            }
    
    def _generate_risk_assessment(self, insights_prep: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment based on complaint themes."""
        main_context = insights_prep.get("prompts_for_llm", {}).get("main_context", "")
        negative_analysis = insights_prep.get("negative_review_analysis", {})
        
        prompt = f"""As a risk analyst, assess the business risks based on these user complaints:

{main_context}

Analyze risks in these areas:

1. **User Retention Risk**
   - Which complaint themes are most likely to cause users to abandon the app?
   - What's the estimated churn risk?

2. **App Store Rating Risk**
   - How might these complaints affect app store ratings and discoverability?
   - What's the reputation risk?

3. **Competitive Risk**
   - Are these complaints about features where competitors excel?
   - What market share risks exist?

4. **Revenue Risk**
   - Which complaints could directly impact monetization?
   - Are there subscription/purchase abandonment risks?

5. **Platform Risk**
   - Any risks related to app store policies or platform changes?

For each risk category, provide:
- Risk level (Critical/High/Medium/Low)
- Potential business impact
- Mitigation strategies
- Early warning indicators to monitor

Be specific about financial and strategic implications."""

        try:
            risk_analysis = self._call_openai(prompt)
            
            return {
                "risk_analysis": risk_analysis,
                "critical_areas": self._identify_critical_risks(negative_analysis),
                "monitoring_recommendations": [
                    "Track app store rating trends weekly",
                    "Monitor negative review percentage monthly", 
                    "Set up alerts for new complaint themes",
                    "Track user retention metrics by cohort",
                    "Monitor competitor app ratings and features"
                ]
            }
            
        except Exception as e:
            return {
                "risk_analysis": f"Error generating risk assessment: {e}",
                "critical_areas": [],
                "monitoring_recommendations": []
            }
    
    def _call_openai(self, prompt: str, max_retries: int = 3) -> str:
        """Call OpenAI API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a senior product analyst and strategist with expertise in mobile app analytics, user experience, and business strategy. Provide detailed, actionable insights based on user review data."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"OpenAI API error (attempt {attempt + 1}): {e}")
                continue
    
    def _calculate_negative_percentage(self, overview: Dict[str, Any]) -> float:
        """Calculate percentage of negative reviews."""
        sentiment = overview.get("sentiment_breakdown", {})
        if not sentiment:
            return 0.0
        
        dist = sentiment.get("distribution", {})
        negative_count = dist.get("negative", {}).get("count", 0)
        total = sentiment.get("total_analyzed", 1)
        
        return round(negative_count / total * 100, 1)
    
    def _get_top_complaint(self, negative_analysis: Dict[str, Any]) -> str:
        """Get the top complaint theme."""
        topics = negative_analysis.get("topic_breakdown", [])
        if not topics:
            return "No complaint themes identified"
        
        # Topics are already sorted by review count in composer
        return topics[0]["topic_name"]
    
    def _extract_priority_areas(self, insights_prep: Dict[str, Any]) -> List[str]:
        """Extract priority areas based on complaint volume."""
        negative_analysis = insights_prep.get("negative_review_analysis", {})
        topics = negative_analysis.get("topic_breakdown", [])
        
        # Return top 3 complaint themes by volume
        return [topic["topic_name"] for topic in topics[:3]]
    
    def _estimate_impact(self, insights_prep: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate potential impact of addressing complaints."""
        negative_analysis = insights_prep.get("negative_review_analysis", {})
        total_negative = negative_analysis.get("total_negative", 0)
        
        if total_negative == 0:
            return {}
        
        # Calculate potential impact if top issues were resolved
        topics = negative_analysis.get("topic_breakdown", [])
        top_3_impact = sum(topic["review_count"] for topic in topics[:3])
        
        return {
            "potential_satisfaction_improvement": f"{round(top_3_impact / total_negative * 100, 1)}%",
            "reviews_that_could_be_improved": top_3_impact,
            "methodology": "Based on addressing top 3 complaint themes"
        }
    
    def _categorize_by_priority(self, negative_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize themes by priority (simplified heuristic)."""
        topics = negative_analysis.get("topic_breakdown", [])
        total_negative = negative_analysis.get("total_negative", 1)
        
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for topic in topics:
            percentage = topic["percentage"]
            review_count = topic["review_count"]
            
            # High priority: >20% of negative reviews or >10 reviews
            if percentage > 20 or review_count > 10:
                high_priority.append(topic["topic_name"])
            # Medium priority: 10-20% or 5-10 reviews  
            elif percentage > 10 or review_count > 5:
                medium_priority.append(topic["topic_name"])
            # Low priority: everything else
            else:
                low_priority.append(topic["topic_name"])
        
        return {
            "high": high_priority,
            "medium": medium_priority,
            "low": low_priority
        }
    
    def _identify_critical_risks(self, negative_analysis: Dict[str, Any]) -> List[str]:
        """Identify critical risk areas."""
        topics = negative_analysis.get("topic_breakdown", [])
        critical_risks = []
        
        for topic in topics:
            # Critical if affects >30% of negative reviews
            if topic["percentage"] > 30:
                critical_risks.append(f"High volume complaints about {topic['topic_name']}")
            
            # Check for specific risk keywords in phrases
            risk_keywords = ["crash", "bug", "broken", "data", "security", "payment", "charge"]
            for phrase in topic["key_phrases"]:
                if any(keyword in phrase.lower() for keyword in risk_keywords):
                    critical_risks.append(f"Technical/Security concerns in {topic['topic_name']}")
                    break
        
        return critical_risks
    
    def print_insights_summary(self, insights: Dict[str, Any]):
        """Print a human-readable summary of generated insights."""
        print("\n" + "="*80)
        print("🧠 AI-GENERATED INSIGHTS SUMMARY")
        print("="*80)
        
        gen_info = insights.get("generation_info", {})
        source_data = gen_info.get("source_data", {})
        
        print(f"Analysis based on {source_data.get('total_reviews', 0)} total reviews")
        print(f"Negative reviews analyzed: {source_data.get('negative_reviews', 0)}")
        print(f"Complaint themes discovered: {source_data.get('topics_discovered', 0)}")
        
        # Executive Summary
        exec_summary = insights.get("executive_summary", {})
        if exec_summary.get("summary_text"):
            print("\n📊 EXECUTIVE SUMMARY:")
            print("-" * 40)
            print(exec_summary["summary_text"])
        
        # Top Recommendations
        recommendations = insights.get("strategic_recommendations", {})
        if recommendations.get("priority_areas"):
            print("\n💡 PRIORITY AREAS:")
            print("-" * 40)
            for i, area in enumerate(recommendations["priority_areas"], 1):
                print(f"{i}. {area}")
        
        # Risk Assessment
        risk_assessment = insights.get("risk_assessment", {})
        if risk_assessment.get("critical_areas"):
            print("\n⚠️  CRITICAL RISKS IDENTIFIED:")
            print("-" * 40)
            for risk in risk_assessment["critical_areas"]:
                print(f"• {risk}")
        
        print("\n" + "="*80)


def generate_insights_from_composition(
    composition_path: Path,
    output_path: Path,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo"
) -> None:
    """
    Generate insights from a composed analysis file.
    """
    generator = ReviewInsightsGenerator(api_key, model)
    
    try:
        # Load composed data
        with open(composition_path, 'r', encoding='utf-8') as f:
            composed_data = json.load(f)
        
        print(f"Loaded composed analysis from {composition_path}")
        
        # Generate insights
        insights = generator.generate_comprehensive_insights(composed_data)
        
        # Save insights
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, ensure_ascii=False, indent=2)
        
        print(f"Saved AI-generated insights to {output_path}")
        
        # Print summary
        generator.print_insights_summary(insights)
        
    except Exception as e:
        raise InsightsError(f"Failed to generate insights from {composition_path}: {e}")


def generate_insights_csv(
    input_path: Path,
    output_path: Path,
    sentiment_col: str = "review_sentiment_class",
    text_col: str = "review_clean_translated",
    min_topic_size: int = 5,
    language_model: str = "all-MiniLM-L6-v2",
    api_key: Optional[str] = None,
    llm_model: str = "gpt-3.5-turbo"
) -> None:
    """
    End-to-end insights generation from CSV file.
    """
    print("🚀 Starting end-to-end insights generation...")
    
    try:
        # Step 1: Compose analysis
        print("Composing analysis...")
        composer = ReviewComposer(min_topic_size, language_model)
        df = pd.read_csv(input_path)
        composed_data = composer.compose_analysis(df, sentiment_col, text_col)
        
        # Step 2: Generate insights
        print("Generating AI insights...")
        generator = ReviewInsightsGenerator(api_key, llm_model)
        insights = generator.generate_comprehensive_insights(composed_data)
        
        # Step 3: Save everything
        print("Saving results...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save composed data
        composed_path = output_path.parent / f"composed_{output_path.stem}.json"
        with open(composed_path, 'w', encoding='utf-8') as f:
            json.dump(composed_data, f, ensure_ascii=False, indent=2)
        
        # Save insights
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, ensure_ascii=False, indent=2)
        
        print(f"Saved composed analysis to {composed_path}")
        print(f"Saved AI insights to {output_path}")
        
        # Print summary
        generator.print_insights_summary(insights)
        
    except Exception as e:
        raise InsightsError(f"Failed to generate insights from {input_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI-powered insights from app review analysis.")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Command 1: Generate from composed file
    compose_parser = subparsers.add_parser('from-composition', help='Generate insights from composed analysis file')
    compose_parser.add_argument("composition", help="Input composed analysis JSON file")
    compose_parser.add_argument("--output", "-o", help="Output insights JSON file (default: insights_<input>)")
    compose_parser.add_argument("--api-key", help="OpenAI API key (default: OPENAI_API_KEY env var)")
    compose_parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model (default: gpt-3.5-turbo)")
    
    # Command 2: End-to-end from CSV
    csv_parser = subparsers.add_parser('from-csv', help='End-to-end insights generation from CSV')
    csv_parser.add_argument("input", help="Input CSV file (with sentiment analysis)")
    csv_parser.add_argument("--output", "-o", help="Output insights JSON file (default: insights_<input>)")
    csv_parser.add_argument("--sentiment-col", default="review_sentiment_class", help="Sentiment column")
    csv_parser.add_argument("--text-col", default="review_clean_translated", help="Text column")
    csv_parser.add_argument("--min-topic-size", type=int, default=5, help="Minimum reviews per topic")
    csv_parser.add_argument("--language-model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    csv_parser.add_argument("--api-key", help="OpenAI API key (default: OPENAI_API_KEY env var)")
    csv_parser.add_argument("--llm-model", default="gpt-3.5-turbo", help="OpenAI model (default: gpt-3.5-turbo)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'from-composition':
            composition_path = Path(args.composition)
            if not composition_path.exists():
                raise InsightsError(f"Composition file not found: {composition_path}")
            
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = composition_path.parent / f"insights_{composition_path.stem}.json"
            
            generate_insights_from_composition(
                composition_path,
                output_path,
                args.api_key,
                args.model
            )
            
        elif args.command == 'from-csv':
            input_path = Path(args.input)
            if not input_path.exists():
                raise InsightsError(f"Input file not found: {input_path}")
            
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = input_path.parent / f"insights_{input_path.stem}.json"
            
            generate_insights_csv(
                input_path,
                output_path,
                args.sentiment_col,
                args.text_col,
                args.min_topic_size,
                args.language_model,
                args.api_key,
                args.llm_model
            )
    
    except InsightsError as e:
        print(f"INSIGHTS ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(2)