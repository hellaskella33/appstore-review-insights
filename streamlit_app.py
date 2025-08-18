"""
Streamlit Web Interface for App Store Review Insights

This provides a user-friendly web interface for analyzing Apple App Store reviews.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Import our pipeline
from pipeline import ReviewAnalysisPipeline


# Page configuration
st.set_page_config(
    page_title="App Store Review Insights",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_id' not in st.session_state:
    st.session_state.analysis_id = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = ReviewAnalysisPipeline()


def create_rating_distribution_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create a rating distribution pie chart."""
    rating_dist = metrics['rating_distribution']['rating_distribution']
    
    ratings = list(rating_dist.keys())
    counts = [rating_dist[r]['count'] for r in ratings]
    colors = ['#FF6B6B', '#FF8E53', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig = go.Figure(data=[go.Pie(
        labels=[f"{r} Star" for r in ratings],
        values=counts,
        hole=0.4,
        marker_colors=colors[:len(ratings)]
    )])
    
    fig.update_layout(
        title="Rating Distribution",
        showlegend=True,
        height=400
    )
    
    return fig


def create_sentiment_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create a sentiment analysis chart."""
    sentiment_dist = metrics['sentiment_distribution']['distribution']
    
    sentiments = list(sentiment_dist.keys())
    counts = [sentiment_dist[s]['count'] for s in sentiments]
    colors = {'positive': '#96CEB4', 'negative': '#FF6B6B', 'neutral': '#FFD93D'}
    
    fig = go.Figure(data=[go.Bar(
        x=sentiments,
        y=counts,
        marker_color=[colors.get(s, '#DDD') for s in sentiments],
        text=counts,
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Sentiment Analysis",
        xaxis_title="Sentiment",
        yaxis_title="Number of Reviews",
        height=400
    )
    
    return fig


def display_analysis_results(results: Dict[str, Any]):
    """Display the analysis results."""
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Reviews",
            results['total_reviews']
        )
    
    with col2:
        st.metric(
            "Negative Reviews",
            results['negative_reviews'],
            f"{results['metrics']['negative_percentage']}%"
        )
    
    with col3:
        avg_rating = results['metrics']['rating_distribution']['average_rating']
        st.metric(
            "Average Rating",
            f"{avg_rating:.2f}/5.0"
        )
    
    with col4:
        st.metric(
            "Topics Discovered",
            results['topics_discovered']
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        rating_chart = create_rating_distribution_chart(results['metrics'])
        st.plotly_chart(rating_chart, use_container_width=True)
    
    with col2:
        sentiment_chart = create_sentiment_chart(results['metrics'])
        st.plotly_chart(sentiment_chart, use_container_width=True)
    
    # Main complaints
    if results['metrics']['main_complaints']:
        st.subheader("🔍 Main Complaint Themes")
        for i, complaint in enumerate(results['metrics']['main_complaints'], 1):
            st.write(f"{i}. {complaint}")
    else:
        st.info("No specific complaint themes identified (may need more negative reviews)")
    
    # Countries analyzed
    st.subheader("🌍 Countries Analyzed")
    countries = results['metrics']['country_breakdown']
    st.write(", ".join(countries))
    
    # Download section
    st.subheader("📥 Download Data")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        if st.button("📄 Download Raw Reviews"):
            analysis_id = results.get('analysis_id', st.session_state.analysis_id)
            if analysis_id:
                st.info(f"Raw reviews available at: `/download/{analysis_id}/raw_reviews`")
    
    with download_col2:
        if st.button("📊 Download Processed Data"):
            analysis_id = results.get('analysis_id', st.session_state.analysis_id)
            if analysis_id:
                st.info(f"Processed data available at: `/download/{analysis_id}/sentiment_reviews`")
    
    with download_col3:
        if st.button("📈 Download Analysis Results"):
            analysis_id = results.get('analysis_id', st.session_state.analysis_id)
            if analysis_id:
                st.info(f"Analysis results available at: `/download/{analysis_id}/composed_analysis`")
    
    # AI Insights (if available)
    if results.get('insights'):
        st.subheader("🧠 AI-Generated Insights")
        with st.expander("View Detailed Insights"):
            st.json(results['insights'])


def main():
    """Main Streamlit application."""
    
    st.title("📱 App Store Review Insights")
    st.markdown("Analyze Apple App Store reviews with AI-powered insights")
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # App ID input
    app_id = st.sidebar.text_input(
        "App Store ID",
        value="835599320",  # TikTok as example
        help="Enter the Apple App Store app ID (numbers only)"
    )
    
    # Countries selection
    available_countries = ['us', 'gb', 'ca', 'au', 'de', 'fr', 'it', 'es', 'jp', 'kr']
    countries = st.sidebar.multiselect(
        "Countries",
        options=available_countries,
        default=['us'],
        help="Select countries to analyze reviews from"
    )
    
    # Reviews per country
    max_reviews = st.sidebar.slider(
        "Max Reviews per Country",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Maximum number of reviews to fetch per country"
    )
    
    # Topic modeling parameters
    min_topic_size = st.sidebar.slider(
        "Minimum Topic Size",
        min_value=3,
        max_value=20,
        value=5,
        help="Minimum number of reviews required to form a topic"
    )
    
    # OpenAI API Key for insights
    openai_key = st.sidebar.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        help="Enter your OpenAI API key for AI-generated insights"
    )
    
    # Analysis section
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Start Analysis", type="primary"):
        if not app_id or not app_id.isdigit():
            st.error("Please enter a valid App Store ID (numbers only)")
            return
        
        if not countries:
            st.error("Please select at least one country")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(message: str):
            status_text.text(message)
        
        try:
            with st.spinner("Running analysis..."):
                pipeline = st.session_state.pipeline
                
                results = pipeline.run_complete_analysis(
                    app_id=app_id,
                    countries=countries,
                    max_reviews_per_country=max_reviews,
                    min_topic_size=min_topic_size,
                    openai_api_key=openai_key if openai_key else None,
                    progress_callback=progress_callback
                )
                
                st.session_state.analysis_results = results
                st.session_state.analysis_id = results['analysis_id']
                progress_bar.progress(1.0)
                status_text.text("Analysis completed!")
                st.success("Analysis completed successfully!")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Analysis failed: {str(e)}")
            return
    
    # Previous analyses section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Previous Analyses")
    
    previous_analyses = st.session_state.pipeline.list_analyses()
    
    if previous_analyses:
        analysis_options = {}
        for analysis in previous_analyses[:10]:  # Show last 10
            timestamp = analysis.get('processed_at', 'Unknown')
            if timestamp and timestamp != 'Unknown':
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    formatted_time = timestamp
            else:
                formatted_time = 'Unknown'
            
            label = f"{formatted_time} ({analysis['total_reviews']} reviews)"
            analysis_options[label] = analysis
        
        selected_analysis = st.sidebar.selectbox(
            "Load Previous Analysis",
            options=[""] + list(analysis_options.keys())
        )
        
        if selected_analysis and selected_analysis in analysis_options:
            analysis_data = analysis_options[selected_analysis]
            analysis_id = analysis_data['analysis_id']
            
            if st.sidebar.button("📊 Load Analysis"):
                full_analysis = st.session_state.pipeline.get_analysis_by_id(analysis_id)
                if full_analysis:
                    st.session_state.analysis_results = full_analysis
                    st.session_state.analysis_id = analysis_id
                    st.success(f"Loaded analysis: {analysis_id}")
                else:
                    st.error("Failed to load analysis")
    else:
        st.sidebar.write("No previous analyses found")
    
    # Main content area
    if st.session_state.analysis_results:
        display_analysis_results(st.session_state.analysis_results)
    else:
        # Welcome message
        st.markdown("""
        ## Welcome! 👋
        
        This tool helps you analyze Apple App Store reviews using AI and machine learning.
        
        ### How to use:
        1. **Enter an App Store ID** in the sidebar (e.g., 835599320 for TikTok)
        2. **Select countries** to analyze reviews from
        3. **Configure analysis parameters** like number of reviews and topic modeling settings
        4. **Optionally add your OpenAI API key** for AI-generated insights
        5. **Click "Start Analysis"** and wait for results
        
        ### Features:
        - 📊 **Rating & Sentiment Analysis**: Understand user sentiment and rating patterns
        - 🎯 **Topic Modeling**: Discover common themes in negative reviews
        - 🌐 **Multi-country Support**: Analyze reviews from different regions
        - 🧠 **AI Insights**: Get intelligent insights powered by OpenAI (optional)
        - 📥 **Data Export**: Download raw and processed data
        
        ### Popular App Store IDs:
        - **TikTok**: 835599320
        - **Instagram**: 389801252
        - **WhatsApp**: 310633997
        - **Netflix**: 363590051
        - **Spotify**: 324684580
        
        Start by entering an App Store ID in the sidebar! 🚀
        """)
        
        # Show example analysis if available
        if previous_analyses:
            st.markdown("### 📈 Recent Analyses")
            
            for analysis in previous_analyses[:3]:
                with st.expander(f"Analysis from {analysis.get('processed_at', 'Unknown')} - {analysis['total_reviews']} reviews"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", analysis['total_reviews'])
                    with col2:
                        st.metric("Negative Reviews", analysis['negative_reviews'])
                    with col3:
                        st.metric("Topics Found", analysis['topics_discovered'])


if __name__ == "__main__":
    main()