import pandas as pd
import streamlit as st
import datetime
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import logging
import email
import re
import html
import ssl
import openai 
import config
from imapclient import IMAPClient
from email.header import decode_header
import json
import plotly.graph_objects as go
import plotly.express as px

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === OpenAI GPT-4o-mini Integration ===
def initialize_openai():
    """Initialize OpenAI client with API key from config."""
    openai.api_key = config.OPENAI_API_KEY

def analyze_sentiment_with_gpt(text):
    """
    Advanced sentiment analysis using OpenAI GPT-4o-mini with comprehensive scoring and tone detection.
    
    Returns:
    - Sentiment Score (1-10)
    - Tone Category
    - Detailed Sentiment Breakdown
    """
    try:
        # Comprehensive sentiment analysis prompt
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an advanced email sentiment and tone analyzer. 
                    Provide a comprehensive analysis with the following details:
                    
                    Sentiment Scoring (1-10 Scale):
                    - 1-4: Negative Sentiments (Highly Dissatisfied to Somewhat Dissatisfied)
                    - 5: Neutral Sentiments
                    - 6-10: Positive Sentiments (Somewhat Satisfied to Highly Satisfied)
                    
                    Tone Categories:
                    Positive: Happy, Excited, Grateful, Hopeful
                    Neutral: Neutral, Curious
                    Negative: Sad, Frustrated, Angry, Disappointed, Confused
                    Complex: Skeptical, Urgent, Apologetic, Overwhelmed
                    Problematic: Sarcastic, Passive-Aggressive, Dismissive, Demanding, Fearful
                    
                    Provide a detailed, structured response."""
                },
                {
                    "role": "user", 
                    "content": f"""Analyze the sentiment and tone of this email text:

                    {text}

                    Please provide:
                    1. Precise Sentiment Score (1-10)
                    2. Primary Tone
                    3. Secondary Tone (if applicable)
                    4. Key Emotional Indicators
                    5. Sentiment Category (Negative/Neutral/Positive)
                    
                    Response Format:
                    Sentiment: [Score]
                    Tone: [Primary Tone]
                    Secondary Tone: [Optional]
                    Category: [Sentiment Category]
                    Indicators: [Key Emotional Words/Phrases]"""
                }
            ],
            max_tokens=250,
            temperature=0.7  # Slight randomness for nuanced analysis
        )
        
        # Extract and parse the analysis
        analysis = response.choices[0].message.content.strip()
        
        # Advanced parsing with multiple extraction attempts
        def extract_value(pattern, default='Unknown'):
            match = re.search(pattern, analysis, re.IGNORECASE)
            return match.group(1).strip() if match else default
        
        # Structured sentiment extraction
        sentiment_score = extract_value(r'Sentiment:\s*(\d+)', '5')
        primary_tone = extract_value(r'Tone:\s*(\w+)', 'Neutral')
        secondary_tone = extract_value(r'Secondary Tone:\s*(\w+)', '')
        sentiment_category = extract_value(r'Category:\s*(\w+)', 'Neutral')
        emotional_indicators = extract_value(r'Indicators:\s*(.+)', '')
        
        # Standardize sentiment score
        try:
            sentiment_score = max(1, min(10, int(sentiment_score)))
        except ValueError:
            sentiment_score = 5
        
        # Comprehensive sentiment description
        sentiment_descriptions = {
            1: "Highly Dissatisfied",
            2: "Very Dissatisfied",
            3: "Dissatisfied",
            4: "Somewhat Dissatisfied",
            5: "Neutral with Concerns",
            6: "Completely Neutral",
            7: "Somewhat Satisfied",
            8: "Satisfied",
            9: "Very Satisfied",
            10: "Highly Satisfied"
        }
        
        # Comprehensive tone categories
        tone_categories = {
            "Positive": ["Happy", "Excited", "Grateful", "Hopeful"],
            "Neutral": ["Neutral", "Curious"],
            "Negative": ["Sad", "Frustrated", "Angry", "Disappointed", "Confused"],
            "Complex": ["Skeptical", "Urgent", "Apologetic", "Overwhelmed"],
            "Problematic": ["Sarcastic", "Passive-Aggressive", "Dismissive", "Demanding", "Fearful"]
        }
        
        # Determine tone category
        def get_tone_category(tone):
            for category, tones in tone_categories.items():
                if tone in tones:
                    return category
            return "Neutral"
        
        tone_category = get_tone_category(primary_tone)
        
        # Construct detailed sentiment result
        detailed_sentiment = {
            "score": sentiment_score,
            "description": sentiment_descriptions[sentiment_score],
            "primary_tone": primary_tone,
            "secondary_tone": secondary_tone,
            "tone_category": tone_category,
            "sentiment_category": sentiment_category,
            "emotional_indicators": emotional_indicators
        }
        
        # Format final sentiment output
        formatted_sentiment = (
            f"Sentiment ({detailed_sentiment['score']}/10): "
            f"{detailed_sentiment['description']} | "
            f"Tone: {detailed_sentiment['primary_tone']} "
            f"(Category: {detailed_sentiment['tone_category']})"
        )
        
        return formatted_sentiment, detailed_sentiment
    
    except Exception as e:
        logging.error(f"Advanced OpenAI Sentiment Analysis Error: {e}")
        return "Neutral (5/10)", {
            "score": 5,
            "description": "Neutral with Concerns",
            "primary_tone": "Neutral",
            "secondary_tone": "",
            "tone_category": "Neutral",
            "sentiment_category": "Neutral",
            "emotional_indicators": ""
        }

# Optional: Visualization function for sentiment analysis
def visualize_sentiment_analysis(detailed_sentiment):
    """
    Create a visual representation of sentiment analysis results.
    Can be used in Streamlit or other visualization contexts.
    """
    import plotly.graph_objects as plt
    
    # Sentiment Score Gauge
    fig = plt.Figure(go.Indicator(
        mode = "gauge+number",
        value = detailed_sentiment['score'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Score"},
        gauge = {
            'axis': {'range': [1, 10]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [1, 4], 'color': "red"},
                {'range': [4, 6], 'color': "yellow"},
                {'range': [6, 10], 'color': "green"}
            ]
        }
    ))
    
    return fig

def generate_reply_draft_with_gpt(text, sentiment, tone):
    """Generate reply draft using OpenAI GPT-4o-mini."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional email response drafter. Create a thoughtful, professional email reply based on the original email's content, sentiment, and tone."},
                {"role": "user", "content": f"Original Email Content:\n{text}\n\nSentiment: {sentiment}\nTone: {tone}\n\nDraft a professional email response that addresses the content, reflects the sentiment, and matches the tone."}
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI Reply Draft Error: {e}")
        return "Unable to generate reply draft. Please draft manually."

def validate_draft_with_gpt(original_email, draft_reply):
    """
    Use GPT to validate the draft reply against the original email
    and provide feedback on completeness and relevance.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional email communication validator. Analyze the original email and proposed reply draft."
                },
                {
                    "role": "user", 
                    "content": f"""
                    Original Email:
                    {original_email}

                    Proposed Reply Draft:
                    {draft_reply}

                    Please provide a analysis in simple terms:
                    1. Did the reply address all key points in the original email?
                    2. Are there any missing critical information or unanswered questions?
                    3. Does the tone and sentiment match the original email?
                    4. Provide a confidence score (0-100) for the draft's effectiveness.
                    5. Suggest specific improvements if needed.

                    Write a final draft:
                    1. write a final draft with proper fomrat with address all queries, explicit request and future request:

                    Format your response clearly with sections for each analysis point.

                    """
                }
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Draft Validation Error: {e}")
        return f"Validation Error: {str(e)}"

def email_reply_drafting_tab(st, session_state):
    st.header("Draft Email Reply: AI-Powered Workflow")
    
    if 'email_data' not in session_state or not session_state['email_data']:
        st.warning("Please extract and analyze emails first.")
        return

    # Step 1: Email Content Selection
    st.subheader("📧 Select Email Content")
    
    # Option to select from previous emails or manual input
    content_source = st.radio(
        "Choose Email Content Source", 
        ["Select from Previous Emails", "Manual Input"],
        key="content_source_selector"
    )
    
    if content_source == "Select from Previous Emails":
        email_df = pd.DataFrame(session_state['email_data'])
        selected_email_index = st.selectbox(
            "Select an Email", 
            range(len(email_df)),
            format_func=lambda x: f"{email_df.iloc[x]['From']} - {email_df.iloc[x]['Subject']}",
            key="previous_email_selection"
        )
        original_email_content = email_df.iloc[selected_email_index]['Body']
    else:
        original_email_content = st.text_area(
            "Enter Email Content", 
            height=200, 
            placeholder="Paste the original email content here...",
            key="manual_email_input"
        )
    
    # Step 2: Reply Draft Generation
    st.subheader("✍️ Reply Draft Generation")
    draft_method = st.radio(
        "Choose Draft Method", 
        ["Auto-Generate AI Draft", "Manual Draft"],
        key="draft_method_selector"
    )
    
    draft_content = ""
    
    if draft_method == "Auto-Generate AI Draft":
        # Automatic AI Draft Generation
        if st.button("Generate AI Draft", key="generate_ai_draft"):
            if original_email_content:
                # Analyze sentiment first
                sentiment, tone = analyze_sentiment_with_gpt(original_email_content)
                
                # Extract requests
                explicit_requests, future_requests = extract_requests(original_email_content)
                
                # Generate AI reply draft with more context
                draft_content = generate_reply_draft_with_gpt(
                    original_email_content, 
                    sentiment, 
                    tone
                )
                
                # Display generated draft immediately
                st.subheader("🤖 AI Generated Draft")
                st.text_area(
                    "Generated Draft", 
                    value=draft_content, 
                    height=250,
                    key="ai_generated_draft"
                )
                
                # Store draft in session state
                session_state['draft_content'] = draft_content
            else:
                st.warning("Please enter email content first.")
    else:
        # Manual Draft Input with inline editing
        draft_content = st.text_area(
            "Write Your Draft Reply", 
            height=250,
            placeholder="Type your manual reply draft here...",
            key="manual_draft_input"
        )
        session_state['draft_content'] = draft_content
    
    # Step 3: Draft Verification (Simplified for Manual Drafts)
    if session_state.get('draft_content'):
        st.subheader("🔍 Draft Verification")
        
        if draft_method == "Manual Draft":
            # Quick verification for manual drafts
            if st.button("Quick Validate", key="quick_validate_button"):
                validation_result = validate_draft_with_gpt(
                    original_email_content, 
                    session_state['draft_content']
                )
                
                # Display concise validation
                st.markdown("### Quick Validation")
                st.markdown(validation_result)
                
                # Correction Input
                correction_input = st.text_area(
                    "Suggest Corrections", 
                    height=100,
                    placeholder="Enter any specific improvements or corrections...",
                    key="draft_correction_input"
                )
                
                # Apply Corrections Button
                if st.button("Apply Corrections", key="apply_corrections_button"):
                    if correction_input:
                        # Use GPT to help refine the draft based on user corrections
                        refined_draft = generate_reply_draft_with_gpt(
                            session_state['draft_content'] + "\n\nUser Suggestions: " + correction_input,
                            "Neutral (5/10)",
                            "Neutral"
                        )
                        
                        # Update draft in session state and UI
                        session_state['draft_content'] = refined_draft
                        st.text_area(
                            "Refined Draft", 
                            value=refined_draft, 
                            height=250,
                            key="refined_draft_output"
                        )
        
        # Finalization Options
        st.subheader("📥 Finalize Draft")
        finalize_choice = st.radio(
            "Draft Status", 
            ["Ready to Send", "Need More Work"],
            key="finalize_draft_radio"
        )
        
        if finalize_choice == "Ready to Send":
            # Final draft download
            st.download_button(
                label="📥 Download Final Draft",
                data=session_state['draft_content'],
                file_name="final_email_reply.txt",
                mime="text/plain",
                key="download_final_draft"
            )

def email_analysis_tab(st, session_state):
    st.header("Analyze Emails")
    
    if 'email_data' not in session_state or not session_state['email_data']:
        st.warning("Please extract emails first in the Extraction tab.")
        return

    # Create a DataFrame from extracted emails
    email_df = pd.DataFrame(session_state['email_data'])
    
    # Select email for detailed analysis
    selected_email_index = st.selectbox(
        "Select an Email for Analysis", 
        range(len(email_df)),
        format_func=lambda x: f"{email_df.iloc[x]['From']} - {email_df.iloc[x]['Subject']}"
    )
    
    # Display selected email details
    selected_email = email_df.iloc[selected_email_index]
    st.subheader("Selected Email Details")
    st.write(f"**From:** {selected_email['From']}")
    st.write(f"**To:** {selected_email['To']}")
    st.write(f"**Subject:** {selected_email['Subject']}")
    st.write(f"**Date:** {selected_email['Date']} {selected_email['Time']}")
    
    # Email body with verification
    st.text_area("Email Body", value=selected_email['Body'], height=200, key="verified_body")
    
    # Perform sentiment analysis
    if st.button("Analyze Sentiment", key="sentiment_analysis"):
        # Use the enhanced sentiment analysis function
        sentiment, detailed_sentiment = analyze_sentiment_with_gpt(selected_email['Body'])
        
        # Create two columns for sentiment details and visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Sentiment Analysis Results")
            st.write(f"**Overall Sentiment:** {sentiment}")
            
            # Display detailed sentiment information
            st.markdown("#### Detailed Breakdown")
            st.write(f"**Score:** {detailed_sentiment['score']}/10")
            st.write(f"**Description:** {detailed_sentiment['description']}")
            st.write(f"**Primary Tone:** {detailed_sentiment['primary_tone']}")
            st.write(f"**Tone Category:** {detailed_sentiment['tone_category']}")
            
            if detailed_sentiment['secondary_tone']:
                st.write(f"**Secondary Tone:** {detailed_sentiment['secondary_tone']}")
            
            if detailed_sentiment['emotional_indicators']:
                st.write("**Emotional Indicators:**")
                st.code(detailed_sentiment['emotional_indicators'])
        
        with col2:
            st.subheader("Sentiment Visualization")
            
            # Sentiment Score Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = detailed_sentiment['score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Score"},
                gauge = {
                    'axis': {'range': [1, 10]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [1, 4], 'color': "red"},
                        {'range': [4, 6], 'color': "yellow"},
                        {'range': [6, 10], 'color': "green"}
                    ]
                }
            ))
            
            # Display the Plotly figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional Visualization: Tone Distribution
            if len(session_state['email_data']) > 1:
                # Analyze tones across all emails
                all_tones = []
                for email in session_state['email_data']:
                    try:
                        _, detailed_analysis = analyze_sentiment_with_gpt(email['Body'])
                        all_tones.append(detailed_analysis['primary_tone'])
                    except Exception as e:
                        logging.error(f"Error analyzing tone: {e}")
                
                tone_counts = pd.Series(all_tones).value_counts()
                
                # Create a pie chart of tone distribution
                tone_fig = px.pie(
                    values=tone_counts.values, 
                    names=tone_counts.index, 
                    title="Tone Distribution Across Emails"
                )
                
                st.plotly_chart(tone_fig, use_container_width=True)

def thread_sentiment_analysis_tab(st, session_state):
    """
    Dedicated tab for comprehensive thread sentiment analysis
    """
    st.header("🧵 Thread Sentiment Deep Dive")
    
    if 'email_data' not in session_state or not session_state['email_data']:
        st.warning("Please extract emails first in the Extraction tab.")
        return

    # Create a DataFrame from extracted emails
    email_df = pd.DataFrame(session_state['email_data'])
    
    # Identify unique threads
    threads = email_df['Thread'].unique()
    
    # Thread selection with additional context
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_thread = st.selectbox(
            "Select a Thread", 
            threads,
            key="thread_sentiment_selector",
            format_func=lambda x: f"{x} (Emails: {len(email_df[email_df['Thread'] == x])})"
        )
    
    with col2:
        # Option to show all threads analysis
        show_all_threads = st.checkbox("Analyze All Threads", key="analyze_all_threads")
    
    # Filter emails for the selected thread
    thread_emails = email_df[email_df['Thread'] == selected_thread]
    
    # Perform thread-level sentiment analysis
    def analyze_thread_sentiment(emails):
        """
        Analyze sentiment for an entire email thread and create visualizations.
        """
        sentiment_results = []
        
        for email in emails.to_dict('records'):
            try:
                # Use GPT-powered sentiment analysis
                sentiment_text, detailed_sentiment = analyze_sentiment_with_gpt(email['Body'])
                
                sentiment_results.append({
                    'From': email['From'],
                    'Date': email['Date'],
                    'Sentiment Score': detailed_sentiment['score'],
                    'Primary Tone': detailed_sentiment['primary_tone'],
                    'Tone Category': detailed_sentiment['tone_category']
                })
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
        
        # Sort results by date
        sentiment_results = sorted(sentiment_results, key=lambda x: x['Date'])
        
        # Visualizations
        def create_thread_sentiment_chart(sentiment_results):
            """Create a line chart showing sentiment scores over time"""
            import plotly.graph_objs as go
            
            # Prepare data for the chart
            dates = [result['Date'] for result in sentiment_results]
            sentiment_scores = [result['Sentiment Score'] for result in sentiment_results]
            senders = [result['From'] for result in sentiment_results]
            primary_tones = [result['Primary Tone'] for result in sentiment_results]
            
            # Create the line chart
            fig = go.Figure()
            
            # Add sentiment score line
            fig.add_trace(go.Scatter(
                x=dates, 
                y=sentiment_scores, 
                mode='lines+markers+text',
                name='Sentiment Score',
                text=[f"{sender}: {tone} ({score}/10)" for sender, tone, score in zip(senders, primary_tones, sentiment_scores)],
                textposition='top center',
                line=dict(color='blue', width=2),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title='Thread Sentiment Progression',
                xaxis_title='Date',
                yaxis_title='Sentiment Score (1-10)',
                yaxis=dict(range=[1, 10])
            )
            
            return fig
        
        def create_tone_distribution_pie(sentiment_results):
            """Create a pie chart showing tone distribution in the thread"""
            import plotly.express as px
            
            # Count tone frequencies
            tone_counts = {}
            for result in sentiment_results:
                tone = result['Primary Tone']
                tone_counts[tone] = tone_counts.get(tone, 0) + 1
            
            # Create pie chart
            tone_df = pd.DataFrame.from_dict(tone_counts, orient='index', columns=['Count']).reset_index()
            tone_df.columns = ['Tone', 'Count']
            
            fig = px.pie(
                tone_df, 
                values='Count', 
                names='Tone', 
                title='Tone Distribution in Thread'
            )
            
            return fig
        
        # Generate visualizations
        thread_sentiment_chart = create_thread_sentiment_chart(sentiment_results)
        tone_distribution_chart = create_tone_distribution_pie(sentiment_results)
        
        # Aggregate thread-level insights
        thread_insights = {
            'Average Sentiment': sum(r['Sentiment Score'] for r in sentiment_results) / len(sentiment_results),
            'Most Common Tone': max(set(r['Primary Tone'] for r in sentiment_results), key=[r['Primary Tone'] for r in sentiment_results].count),
            'Sentiment Trend': 'Positive' if sum(r['Sentiment Score'] for r in sentiment_results) / len(sentiment_results) > 6 else 'Negative'
        }
        
        return {
            'sentiment_results': sentiment_results,
            'thread_insights': thread_insights,
            'charts': {
                'sentiment_chart': thread_sentiment_chart,
                'tone_distribution': tone_distribution_chart
            }
        }
    
    # Perform analysis based on selection
    if show_all_threads:
        # Analysis for all threads
        st.subheader("Multi-Thread Sentiment Overview")
        
        # Create a summary dataframe of thread sentiments
        thread_summaries = []
        
        for thread in threads:
            thread_df = email_df[email_df['Thread'] == thread]
            thread_analysis = analyze_thread_sentiment(thread_df)
            
            thread_summaries.append({
                'Thread': thread,
                'Avg Sentiment': thread_analysis['thread_insights']['Average Sentiment'],
                'Most Common Tone': thread_analysis['thread_insights']['Most Common Tone'],
                'Sentiment Trend': thread_analysis['thread_insights']['Sentiment Trend'],
                'Email Count': len(thread_df)
            })
        
        # Display thread summaries
        summary_df = pd.DataFrame(thread_summaries)
        st.dataframe(summary_df, use_container_width=True)
        
        # Optional: Visualize thread sentiments
        if st.checkbox("Visualize Thread Sentiments", key="visualize_all_threads"):
            # Bar chart of thread sentiments
            import plotly.express as px
            
            fig = px.bar(
                summary_df, 
                x='Thread', 
                y='Avg Sentiment', 
                color='Sentiment Trend',
                title='Average Sentiment Across Threads',
                labels={'Avg Sentiment': 'Average Sentiment Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Analysis for selected thread
        st.subheader(f"Thread Sentiment Analysis: {selected_thread}")
        
        # Display thread emails
        st.dataframe(thread_emails[['From', 'To', 'Date', 'Time', 'Subject']], use_container_width=True)
        
        # Perform thread-level sentiment analysis
        thread_analysis = analyze_thread_sentiment(thread_emails)
        
        # Display thread insights
        st.subheader("Thread Sentiment Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Sentiment", 
                      f"{thread_analysis['thread_insights']['Average Sentiment']:.2f}/10")
        
        with col2:
            st.metric("Most Common Tone", 
                      thread_analysis['thread_insights']['Most Common Tone'])
        
        with col3:
            st.metric("Overall Sentiment Trend", 
                      thread_analysis['thread_insights']['Sentiment Trend'])
        
        # Visualizations
        st.subheader("Sentiment Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(thread_analysis['charts']['sentiment_chart'], use_container_width=True)
        
        with col2:
            st.plotly_chart(thread_analysis['charts']['tone_distribution'], use_container_width=True)
        
        # Detailed sentiment results
        st.subheader("Detailed Sentiment Results")
        st.dataframe(pd.DataFrame(thread_analysis['sentiment_results']), use_container_width=True)

def process_emails(mail, email_ids, max_emails=None):
    """Process emails with progress tracking and optional limits."""
    if max_emails and max_emails < len(email_ids):
        email_ids = email_ids[:max_emails]
    
    progress_bar = st.progress(0)
    email_data = []
    
    for i, eid in enumerate(email_ids):
        # Update progress
        progress = (i + 1) / len(email_ids)
        progress_bar.progress(progress)
        
        try:
            # Parse the email
            rows = parse_email(mail, eid)
            email_data.extend(rows)
        
        except Exception as email_error:
            st.error(f"Error processing email {eid}: {email_error}")
        
        # Update status
        st.sidebar.text(f"Processing: {i+1}/{len(email_ids)}")
    
    progress_bar.empty()
    
    return email_data

def connect_to_email(username, password):
    """Connect to IMAP server via IMAPClient with SSL verification disabled (TEMPORARY)."""
    try:
        # Create SSL context with SSL verification disabled
        context = ssl.create_default_context()
        context.check_hostname = False  # Disable hostname check
        context.verify_mode = ssl.CERT_NONE  # Disable certificate verification

        # Connect to IMAP server using custom SSL context
        mail = IMAPClient(config.IMAP_SERVER, ssl=True, ssl_context=context)
        mail.login(username, password)
        return mail
    except Exception as e:
        logger.error(f"Error connecting to email: {e}")
        return None

def search_emails(mail, subject, date_since, folder="INBOX"):
    """Search for emails based on subject and date range with better error handling."""
    try:
        mail.select_folder(folder)
        messages = mail.search(["SINCE", date_since, "SUBJECT", subject])
        logger.info(f"Found {len(messages)} emails matching criteria")
        return messages
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

# === Helper Functions ===
def simple_sentence_split(text):
    """Split text into sentences using regex patterns."""
    # Common sentence ending patterns
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    return re.split(pattern, text)

def decode_text(text, encoding=None):
    """Helper function to decode text with proper error handling."""
    if isinstance(text, bytes):
        return text.decode(encoding or "utf-8", errors="replace")
    return text

def parse_email_header(header_value):
    """Parse and clean email header values properly."""
    if not header_value:
        return "Unknown"
    
    decoded_parts = decode_header(header_value)
    cleaned_value = ""
    
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            cleaned_value += part.decode(encoding or "utf-8", errors="replace")
        else:
            cleaned_value += str(part)
            
    return cleaned_value.strip()

def parse_date(date_str):
    """Parse email date string with support for multiple formats."""
    if not date_str:
        return {"date": "Unknown Date", "time": "Unknown Time"}
    
    # List of common email date formats
    date_formats = [
        "%a, %d %b %Y %H:%M:%S %z",  # Standard format
        "%d %b %Y %H:%M:%S %z",      # Alternative format
        "%a, %d %b %Y %H:%M:%S",     # No timezone
        "%d %b %Y %H:%M:%S"          # Simplified
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.datetime.strptime(date_str, fmt)
            return {
                "date": parsed_date.strftime("%Y-%m-%d"), 
                "time": parsed_date.strftime("%H:%M:%S")
            }
        except ValueError:
            continue
    
    # If all formats fail, try a regex approach
    try:
        # Extract date components using regex
        match = re.search(r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})', date_str)
        if match:
            day, month, year = match.groups()
            time_match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', date_str)
            time_str = "00:00:00"
            if time_match:
                hour, minute, second = time_match.groups()
                time_str = f"{hour.zfill(2)}:{minute}:{second}"
            
            months = {
                "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
                "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
            }
            month_num = months.get(month, "01")
            return {
                "date": f"{year}-{month_num}-{day.zfill(2)}",
                "time": time_str
            }
    except Exception as e:
        logger.warning(f"Date parsing error: {e}")
    
    return {"date": "Unknown Date", "time": "Unknown Time"}

def get_email_body(msg):
    """Extract email body with better multipart handling."""
    body = ""
    
    if msg.is_multipart():
        # First try to find text/plain part
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = decode_text(payload)
                    break
        
        # If no text/plain, try text/html as fallback
        if not body:
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if "attachment" in content_disposition:
                    continue
                    
                if content_type == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        # Convert HTML to plain text (simplified)
                        html_text = decode_text(payload)
                        body = re.sub('<[^<]+?>', ' ', html_text)  # Remove HTML tags
                        body = re.sub(r'\s+', ' ', body)  # Normalize whitespace
                        break
    else:
        # Not multipart - get payload directly
        payload = msg.get_payload(decode=True)
        if payload:
            body = decode_text(payload)
    
    return body

def extract_clean_body(msg):
    """Extract and clean email body with better forwarded email handling."""
    body = get_email_body(msg)
    
    # Normalize line endings
    body = re.sub(r'\r\n', '\n', body)
    
    # HTML escape content for safety
    body = html.escape(body)
    
    # Clean the body and extract forwarded messages
    cleaned_body, forwarded_emails = parse_email_threads(body)
    
    return cleaned_body, forwarded_emails

def remove_signatures_and_disclaimers(text):
    """Remove email signatures, disclaimers, and other noise."""
    # List of signature markers
    signature_markers = [
        r"(?i)Best Regards,",
        r"(?i)Thanks & Regards,", 
        r"(?i)Kind Regards,",
        r"(?i)Regards,",
        r"(?i)Sincerely,",
        r"(?i)Cheers,",
        r"(?i)Thank you,",
        r"(?i)Thanks,",
        r"--\s*\n",  # Common signature separator
        r"(?i)This email and any files transmitted",
        r"(?i)CONFIDENTIAL",
        r"(?i)DISCLAIMER",
        r"(?i)LEGAL NOTICE",
        r"(?i)PRIVILEGED AND CONFIDENTIAL",
        r"(?i)The information contained in this email",
        r"(?i)This message contains confidential information",
        r"(?i)Do not disseminate",
        r"(?i)Delete this e-mail",
        r"(?i)Intended solely",
        r"(?i)M:\s*\+\d+",  # Mobile numbers
        r"(?i)T:\s*\+\d+",  # Phone numbers
        r"(?i)LinkedIn:",
        r"(?i)Twitter:",
        r"(?i)Facebook:",
        r"(?i)Sent from my iPhone",
        r"(?i)Sent from my Android",
    ]
    
    # Split text into sentences using our simple function
    sentences = simple_sentence_split(text)
    filtered_sentences = []
    
    # Flag to skip lines after a signature marker is found
    skip_remaining = False
    
    for sentence in sentences:
        if skip_remaining:
            # Check if this might be a new section that should be kept
            if re.match(r'^(From|To|Subject|Date):', sentence.strip()):
                skip_remaining = False
            else:
                continue
                
        # Check if this sentence contains a signature marker
        contains_marker = False
        for marker in signature_markers:
            if re.search(marker, sentence):
                contains_marker = True
                skip_remaining = True
                break
                
        if not contains_marker:
            filtered_sentences.append(sentence)
    
    cleaned_text = ' '.join(filtered_sentences)
    
    # Remove redundant whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def parse_email_threads(body):
    """Parse email thread with improved forwarded message handling."""
    # Look for common forwarded and reply patterns
    forwarded_patterns = [
        r'-{3,}\s*Forwarded message\s*-{3,}',
        r'Begin forwarded message:',
        r'---------- Original Message ----------',
        r'From:.*\nDate:.*\nSubject:.*\nTo:',
    ]
    
    # Join all patterns with OR operator
    combined_pattern = '|'.join(forwarded_patterns)
    
    # Split the email body
    thread_parts = re.split(f'({combined_pattern})', body, flags=re.IGNORECASE)
    
    # The main email body is the first part
    main_body = thread_parts[0].strip()
    main_body = remove_signatures_and_disclaimers(main_body)
    
    # Process forwarded parts
    forwarded_emails = []
    current_part = ""
    capture_mode = False
    
    for i in range(1, len(thread_parts)):
        part = thread_parts[i].strip()
        
        # If this is a forwarded message delimiter
        if re.search(combined_pattern, part, re.IGNORECASE):
            capture_mode = True
            current_part = part
        elif capture_mode:
            # Extract forwarded email details
            current_part += "\n" + part
            forwarded_info = extract_forwarded_details(current_part)
            
            if forwarded_info:
                # Clean the forwarded content
                forwarded_info["Content"] = remove_signatures_and_disclaimers(forwarded_info["Content"])
                forwarded_emails.append(forwarded_info)
            
            capture_mode = False
            current_part = ""
    
    return main_body, forwarded_emails

def extract_forwarded_details(section):
    """Extract details from forwarded email with improved parsing."""
    section = section.strip()
    
    # Extract email headers
    from_match = re.search(r'(?i)From:\s*(.*?)(?:\n|$)', section)
    to_match = re.search(r'(?i)To:\s*(.*?)(?:\n|$)', section)
    date_match = re.search(r'(?i)Date:\s*(.*?)(?:\n|$)', section)
    subject_match = re.search(r'(?i)Subject:\s*(.*?)(?:\n|$)', section)
    
    from_email = from_match.group(1).strip() if from_match else "Unknown"
    to_email = to_match.group(1).strip() if to_match else "Unknown"
    raw_date = date_match.group(1).strip() if date_match else None
    subject = subject_match.group(1).strip() if subject_match else "Unknown"
    
    date_info = parse_date(raw_date)
    
    # Remove headers to extract content
    content_pattern = r'(?i)(From|Date|To|Cc|Subject|Sent):.+?(?=\n\n|\n[A-Za-z])'
    content = re.sub(content_pattern, '', section, flags=re.DOTALL)
    
    # Remove remaining header-like parts
    content = re.sub(r'(?i)-{3,}\s*Forwarded message\s*-{3,}', '', content)
    content = re.sub(r'(?i)Begin forwarded message:', '', content)
    content = re.sub(r'(?i)---------- Original Message ----------', '', content)
    
    # Clean up whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()
    
    return {
        "From": from_email,
        "To": to_email,
        "Date": date_info["date"],
        "Time": date_info["time"],
        "Subject": subject,
        "Content": content
    }

def parse_email(mail, email_id):
    """Parse an email with improved data extraction."""
    try:
        raw_message = mail.fetch([email_id], ["RFC822"])[email_id][b"RFC822"]
        msg = email.message_from_bytes(raw_message)
        
        # Extract clean header fields
        subject = parse_email_header(msg["Subject"])
        sender = parse_email_header(msg["From"])
        receiver = parse_email_header(msg["To"])
        
        # Parse date properly
        date_str = msg.get("Date")
        date_time = parse_date(date_str)
        
        # Extract body and forwarded messages
        main_body, forwarded_emails = extract_clean_body(msg)
        
        # Create the main email entry
        email_rows = [{
            "From": sender,
            "To": receiver,
            "Date": date_time["date"],
            "Time": date_time["time"],
            "Subject": subject,
            "Body": main_body,
            "Thread": "Main Email"
        }]
        
        # Add forwarded emails as separate rows
        for idx, forward in enumerate(forwarded_emails, 1):
            email_rows.append({
                "From": forward["From"],
                "To": forward["To"],
                "Date": forward["Date"],
                "Time": forward["Time"],
                "Subject": subject,  # Keep original subject for threading
                "Body": forward["Content"],
                "Thread": f"Forwarded {idx}"
            })
        
        return email_rows
    
    except Exception as e:
        logger.error(f"Error parsing email {email_id}: {e}")
        return [{
            "From": "Error",
            "To": "Error",
            "Date": "Unknown Date",
            "Time": "Unknown Time",
            "Subject": f"Error parsing email {email_id}",
            "Body": f"Error: {str(e)}",
            "Thread": "Error"
        }]

def save_as_table(emails, output_file=None):
    """Convert email data to DataFrame with improved handling."""
    if not emails:
        return pd.DataFrame()
        
    df = pd.DataFrame(emails)
    
    # Fill any missing values
    df = df.fillna("N/A")
    
    # Sort by Date and Time
    try:
        # Convert to datetime for proper sorting
        df["SortDate"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
        df = df.sort_values(by=["SortDate"], ascending=True)
        df = df.drop(columns=["SortDate"])
    except Exception as e:
        logger.warning(f"Error sorting by date: {e}")
        # Fallback: sort by string comparison
        df = df.sort_values(by=["Date", "Time"], ascending=True)
    
    # Save to file if requested
    if output_file:
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Saved data to {output_file}")
    
    return df

def extract_requests(text):
    """Extract explicit and future requests from the text."""
    explicit_requests = []
    future_requests = []

    # Keywords for explicit requests
    explicit_keywords = ["please", "kindly", "request", "send", "provide", "need", "require"]
    # Keywords for future requests
    future_keywords = ["will need", "will require", "by next", "by tomorrow", "soon", "later"]

    # Split text into sentences
    sentences = simple_sentence_split(text)

    for sentence in sentences:
        # Check for explicit requests
        if any(keyword in sentence.lower() for keyword in explicit_keywords):
            explicit_requests.append(sentence.strip())
        # Check for future requests
        if any(keyword in sentence.lower() for keyword in future_keywords):
            future_requests.append(sentence.strip())

    return explicit_requests, future_requests

# === STREAMLIT UI ===
def main():
    st.set_page_config(page_title="📨 AI Email Assistant", layout="wide")
    
    # Initialize OpenAI
    initialize_openai()

    st.title("🤖 AI Email Sentiment & Reply Assistant")

    # Ensure session state for storing extracted emails
    if "email_data" not in st.session_state:
        st.session_state["email_data"] = None

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📬 Email Extraction", "🔍 Email Analysis", "✉️ Reply Drafting", "🧵 Thread Sentiment"])

    # === EMAIL EXTRACTION TAB ===
    with tab1:
        st.header("Extract Emails")
        
        # Sidebar for email connection details
        with st.sidebar:
            st.subheader("Email Connection")
            use_default = st.checkbox("Use default credentials", value=True, key="use_default_credentials")
            
            if use_default:
                email_user = config.DEFAULT_EMAIL
                email_pass = config.DEFAULT_PASSWORD
                st.success("Using saved credentials")
            else:
                email_user = st.text_input("Email", type="default", key="email_user_input")
                email_pass = st.text_input("Password", type="password", key="email_password_input")
            
            # Search parameters
            subject_query = st.text_input("Subject Keywords", "739490")
            date_range = st.date_input(
                "Date Range",
                value=(
                    datetime.date.today() - datetime.timedelta(days=90),
                    datetime.date.today()
                )
            )
            
            folder = st.selectbox(
                "Email Folder",
                ["INBOX", "Sent", "Drafts", "Archive", "Trash"],
                index=0
            )
            
            max_emails = st.number_input(
                "Max Emails to Process",
                min_value=1,
                max_value=1000,
                value=100
            )

            filter_options = st.multiselect(
                "Additional Filters",
                ["Remove Signatures", "Remove Disclaimers", "Remove Links", "Only Main Content"],
                default=["Remove Signatures", "Remove Disclaimers"]
            )
        
        # Extraction button
        extract_btn = st.button("🔍 Extract Emails", key="extract_emails")
        
        if extract_btn:
            with st.spinner("Connecting to email server..."):
                mail = connect_to_email(email_user, email_pass)
            
            if mail:
                with st.spinner(f"Searching for emails with subject '{subject_query}'..."):
                    email_ids = search_emails(
                        mail,
                        subject_query,
                        date_range[0].strftime("%d-%b-%Y"),
                        folder=folder
                    )

                if email_ids:
                    st.success(f"Found {len(email_ids)} matching emails!")

                    with st.spinner("Processing emails..."):
                        email_data = process_emails(
                            mail,
                            email_ids,
                            max_emails=max_emails
                        )

                    # Store email data in session state
                    st.session_state['email_data'] = email_data
                    
                    # Display extracted emails
                    st.dataframe(
                        pd.DataFrame(email_data),
                        use_container_width=True
                    )

                    # Close the connection
                    mail.logout()
                else:
                    st.warning("⚠️ No emails found matching your criteria")
            else:
                st.error("Failed to connect to email server. Check credentials.")

    # === EMAIL ANALYSIS TAB ===
    with tab2:
        email_analysis_tab(st, st.session_state)

    # === REPLY DRAFTING TAB ===
    with tab3:
        email_reply_drafting_tab(st, st.session_state)

    # === THREAD SENTIMENT TAB ===
    with tab4:
        thread_sentiment_analysis_tab(st, st.session_state)

if __name__ == "__main__":
    main()