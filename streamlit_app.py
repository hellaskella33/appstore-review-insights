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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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


def create_topics_chart(results: Dict[str, Any]) -> go.Figure:
    """Create a chart showing topic distribution."""
    if not results.get('metrics', {}).get('main_complaints'):
        return None
    
    # This is a placeholder - we'll need to extract topic data from composed analysis
    # For now, create a simple bar chart of main complaints
    complaints = results['metrics']['main_complaints']
    if not complaints:
        return None
    
    fig = go.Figure(data=[go.Bar(
        x=[f"Topic {i+1}" for i in range(len(complaints))],
        y=[100-i*10 for i in range(len(complaints))],  # Placeholder values
        marker_color='#4ECDC4',
        text=[f"Topic {i+1}" for i in range(len(complaints))],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Topic Distribution",
        xaxis_title="Topics",
        yaxis_title="Number of Reviews",
        height=400
    )
    
    return fig


def create_rating_vs_sentiment_chart(results: Dict[str, Any]) -> go.Figure:
    """Create a chart comparing ratings vs sentiment."""
    rating_dist = results['metrics']['rating_distribution']['rating_distribution']
    sentiment_dist = results['metrics']['sentiment_distribution']['distribution']
    
    ratings = list(rating_dist.keys())
    rating_counts = [rating_dist[r]['count'] for r in ratings]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Rating Distribution', 'Sentiment vs Rating Correlation'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Rating distribution
    fig.add_trace(
        go.Bar(x=ratings, y=rating_counts, name="Ratings", marker_color='#45B7D1'),
        row=1, col=1
    )
    
    # Sentiment vs Rating correlation (simplified)
    sentiment_names = list(sentiment_dist.keys())
    sentiment_counts = [sentiment_dist[s]['count'] for s in sentiment_names]
    colors = {'positive': '#96CEB4', 'negative': '#FF6B6B', 'neutral': '#FFD93D'}
    
    fig.add_trace(
        go.Scatter(
            x=sentiment_names,
            y=sentiment_counts,
            mode='markers+lines',
            name="Sentiment",
            marker=dict(
                size=15,
                color=[colors.get(s, '#DDD') for s in sentiment_names]
            )
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_insights_priority_chart(insights: Dict[str, Any]) -> go.Figure:
    """Create a priority matrix visualization."""
    if not insights.get('priority_matrix', {}).get('themes_by_priority'):
        return None
    
    themes = insights['priority_matrix']['themes_by_priority']
    
    priorities = []
    counts = []
    colors = []
    
    if themes.get('high'):
        priorities.extend(['High Priority'] * len(themes['high']))
        counts.extend([3] * len(themes['high']))  # Weight high priority items
        colors.extend(['#FF6B6B'] * len(themes['high']))
    
    if themes.get('medium'):
        priorities.extend(['Medium Priority'] * len(themes['medium']))
        counts.extend([2] * len(themes['medium']))
        colors.extend(['#FFD93D'] * len(themes['medium']))
    
    if themes.get('low'):
        priorities.extend(['Low Priority'] * len(themes['low']))
        counts.extend([1] * len(themes['low']))
        colors.extend(['#96CEB4'] * len(themes['low']))
    
    if not priorities:
        return None
    
    fig = go.Figure(data=[go.Bar(
        x=['High Priority', 'Medium Priority', 'Low Priority'],
        y=[
            len(themes.get('high', [])),
            len(themes.get('medium', [])),
            len(themes.get('low', []))
        ],
        marker_color=['#FF6B6B', '#FFD93D', '#96CEB4'],
        text=[
            len(themes.get('high', [])),
            len(themes.get('medium', [])),
            len(themes.get('low', []))
        ],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Issue Priority Distribution",
        xaxis_title="Priority Level",
        yaxis_title="Number of Issues",
        height=400
    )
    
    return fig


def create_comprehensive_dashboard(results: Dict[str, Any]) -> go.Figure:
    """Create a comprehensive dashboard with multiple metrics."""
    metrics = results['metrics']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Rating Distribution', 
            'Sentiment Analysis', 
            'Negative Review Breakdown',
            'Review Volume by Country'
        ),
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ]
    )
    
    # Rating distribution (pie chart)
    rating_dist = metrics['rating_distribution']['rating_distribution']
    ratings = list(rating_dist.keys())
    counts = [rating_dist[r]['count'] for r in ratings]
    
    fig.add_trace(
        go.Pie(
            labels=[f"{r} Star" for r in ratings],
            values=counts,
            name="Ratings"
        ),
        row=1, col=1
    )
    
    # Sentiment analysis (bar chart)
    sentiment_dist = metrics['sentiment_distribution']['distribution']
    sentiments = list(sentiment_dist.keys())
    sentiment_counts = [sentiment_dist[s]['count'] for s in sentiments]
    colors = {'positive': '#96CEB4', 'negative': '#FF6B6B', 'neutral': '#FFD93D'}
    
    fig.add_trace(
        go.Bar(
            x=sentiments,
            y=sentiment_counts,
            marker_color=[colors.get(s, '#DDD') for s in sentiments],
            name="Sentiment"
        ),
        row=1, col=2
    )
    
    # Negative review breakdown
    negative_pct = metrics['negative_percentage']
    positive_pct = 100 - negative_pct
    
    fig.add_trace(
        go.Bar(
            x=['Positive/Neutral', 'Negative'],
            y=[positive_pct, negative_pct],
            marker_color=['#96CEB4', '#FF6B6B'],
            name="Review Sentiment %"
        ),
        row=2, col=1
    )
    
    # Countries (if multiple)
    countries = metrics['country_breakdown']
    if len(countries) > 1:
        # Placeholder - equal distribution for now
        country_counts = [results['total_reviews'] // len(countries)] * len(countries)
        fig.add_trace(
            go.Bar(
                x=countries,
                y=country_counts,
                marker_color='#4ECDC4',
                name="Reviews by Country"
            ),
            row=2, col=2
        )
    else:
        fig.add_trace(
            go.Bar(
                x=[countries[0]],
                y=[results['total_reviews']],
                marker_color='#4ECDC4',
                name="Total Reviews"
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title=dict(
            text="Comprehensive Analysis Dashboard",
            x=0.5,
            font=dict(size=20, family="Arial Black", color="#2E86AB")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=11),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update individual subplot styling
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    
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
    
    analysis_id = results.get('analysis_id', st.session_state.analysis_id)
    pipeline = st.session_state.pipeline
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        # Raw Reviews Download
        raw_file_path = pipeline.get_file_path(analysis_id, 'raw_reviews') if analysis_id else None
        if raw_file_path and raw_file_path.exists():
            with open(raw_file_path, 'rb') as file:
                st.download_button(
                    label="📄 Download Raw Reviews",
                    data=file.read(),
                    file_name=f"raw_reviews_{analysis_id}.csv",
                    mime="text/csv"
                )
        else:
            st.button("📄 Raw Reviews (Not Available)", disabled=True)
    
    with download_col2:
        # Processed Data Download
        sentiment_file_path = pipeline.get_file_path(analysis_id, 'sentiment_reviews') if analysis_id else None
        if sentiment_file_path and sentiment_file_path.exists():
            with open(sentiment_file_path, 'rb') as file:
                st.download_button(
                    label="📊 Download Processed Data",
                    data=file.read(),
                    file_name=f"processed_reviews_{analysis_id}.csv",
                    mime="text/csv"
                )
        else:
            st.button("📊 Processed Data (Not Available)", disabled=True)
    
    with download_col3:
        # Analysis Results Download
        composed_file_path = pipeline.get_file_path(analysis_id, 'composed_analysis') if analysis_id else None
        if composed_file_path and composed_file_path.exists():
            with open(composed_file_path, 'rb') as file:
                st.download_button(
                    label="📈 Download Analysis Results",
                    data=file.read(),
                    file_name=f"analysis_results_{analysis_id}.json",
                    mime="application/json"
                )
        else:
            st.button("📈 Analysis Results (Not Available)", disabled=True)
    
    # Additional download for AI Insights if available
    if results.get('insights'):
        insights_file_path = pipeline.get_file_path(analysis_id, 'insights') if analysis_id else None
        if insights_file_path and insights_file_path.exists():
            with open(insights_file_path, 'rb') as file:
                st.download_button(
                    label="🧠 Download AI Insights",
                    data=file.read(),
                    file_name=f"ai_insights_{analysis_id}.json",
                    mime="application/json"
                )
    
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
    
    # App search method
    search_method = st.sidebar.radio(
        "Search Method",
        ["App Store ID", "App Name"],
        help="Choose how to identify the app"
    )
    
    if search_method == "App Store ID":
        app_id = st.sidebar.text_input(
            "App Store ID",
            value="835599320",  # TikTok as example
            help="Enter the Apple App Store app ID (numbers only)"
        )
        app_name = None
    else:
        app_name = st.sidebar.text_input(
            "App Name",
            value="TikTok",  # TikTok as example
            help="Enter the app name to search for"
        )
        app_id = None
    
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
    
    
    # Analysis section
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Start Analysis", type="primary"):
        if search_method == "App Store ID":
            if not app_id or not app_id.isdigit():
                st.error("Please enter a valid App Store ID (numbers only)")
                return
        else:
            if not app_name or not app_name.strip():
                st.error("Please enter an app name")
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
                    app_id=app_id if search_method == "App Store ID" else None,
                    app_name=app_name if search_method == "App Name" else None,
                    countries=countries,
                    max_reviews_per_country=max_reviews,
                    min_topic_size=min_topic_size,
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
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
            
            app_name = analysis.get('app_name', 'Unknown App')
            label = f"{app_name} - {formatted_time} ({analysis['total_reviews']} reviews)"
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
        st.info("Configure analysis settings in the sidebar and click 'Start Analysis' to begin.")
        
        # Show example analysis if available
        if previous_analyses:
            st.markdown("### 📈 Recent Analyses")
            
            for analysis in previous_analyses[:3]:
                app_name = analysis.get('app_name', 'Unknown App')
                timestamp = analysis.get('processed_at', 'Unknown')
                if timestamp and timestamp != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_time = timestamp
                else:
                    formatted_time = 'Unknown'
                
                with st.expander(f"{app_name} - {formatted_time} ({analysis['total_reviews']} reviews)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", analysis['total_reviews'])
                    with col2:
                        st.metric("Negative Reviews", analysis['negative_reviews'])
                    with col3:
                        st.metric("Topics Found", analysis['topics_discovered'])


if __name__ == "__main__":
    main()