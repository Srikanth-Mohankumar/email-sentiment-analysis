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
import config  # Import credentials from config.py
from imapclient import IMAPClient
from email.header import decode_header

st.set_page_config(page_title="📨 Email Sentiment Analysis", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Load Pre-Trained Model Once (No Redownload) ===
@st.cache_resource
def load_model():
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Define 10-category sentiment labels
sentiment_labels = [
    "Highly Dissatisfied (1)", "Very Dissatisfied (2)", "Dissatisfied (3)",
    "Somewhat Dissatisfied (4)", "Neutral with Concerns (5)", "Completely Neutral (6)",
    "Somewhat Satisfied (7)", "Satisfied (8)", "Very Satisfied (9)", "Highly Satisfied (10)"
]

# === Sentiment Analysis Function ===
def analyze_sentiment(text):
    if not text or not isinstance(text, str):
        return "Unknown", "Unknown"  # Handle missing data
    
    # Sentiment analysis (existing logic)
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    
    scores = softmax(outputs.logits.numpy())[0]
    max_index = torch.argmax(outputs.logits).item()

    if max_index == 0:  # Negative
        sentiment = sentiment_labels[0]
    elif max_index == 1:  # Neutral
        sentiment = sentiment_labels[5]
    else:  # Positive
        sentiment = sentiment_labels[9]

    # Tone analysis (new logic)
    tones = [
        "Happy", "Excited", "Grateful", "Neutral", "Confused", "Sad", "Frustrated", 
        "Angry", "Disappointed", "Sarcastic", "Passive-Aggressive", "Urgent", 
        "Hopeful", "Skeptical", "Demanding", "Apologetic", "Fearful", "Curious", 
        "Overwhelmed", "Dismissive"
    ]

    # Simple rule-based tone classification (can be replaced with a model)
    tone = "Neutral"  # Default tone
    if "thank" in text.lower() or "grateful" in text.lower():
        tone = "Grateful"
    elif "urgent" in text.lower() or "immediately" in text.lower():
        tone = "Urgent"
    elif "sorry" in text.lower() or "apologize" in text.lower():
        tone = "Apologetic"
    elif "happy" in text.lower() or "excited" in text.lower():
        tone = "Happy"
    elif "frustrated" in text.lower() or "angry" in text.lower():
        tone = "Frustrated"
    elif "confused" in text.lower() or "unsure" in text.lower():
        tone = "Confused"
    elif "hope" in text.lower() or "wish" in text.lower():
        tone = "Hopeful"
    elif "demand" in text.lower() or "require" in text.lower():
        tone = "Demanding"
    elif "sad" in text.lower() or "disappointed" in text.lower():
        tone = "Sad"
    elif "sarcastic" in text.lower() or "passive-aggressive" in text.lower():
        tone = "Sarcastic"
    elif "fear" in text.lower() or "worried" in text.lower():
        tone = "Fearful"
    elif "curious" in text.lower() or "wonder" in text.lower():
        tone = "Curious"
    elif "overwhelmed" in text.lower() or "too much" in text.lower():
        tone = "Overwhelmed"
    elif "dismiss" in text.lower() or "ignore" in text.lower():
        tone = "Dismissive"

    return sentiment, tone

# === Email Processing Functions ===
logger = logging.getLogger(__name__)

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
        
        # Parse the email
        rows = parse_email(mail, eid)
        for row in rows:
            # Analyze sentiment and tone
            sentiment, tone = analyze_sentiment(row["Body"])
            # Extract requests
            explicit_requests, future_requests = extract_requests(row["Body"])
            # Generate reply draft
            reply_draft = generate_reply_draft(sentiment, tone, explicit_requests, future_requests)
            
            # Add new fields to the row
            row["Sentiment"] = sentiment
            row["Tone"] = tone
            row["Explicit Requests"] = ", ".join(explicit_requests)
            row["Future Requests"] = ", ".join(future_requests)
            row["Reply Draft"] = reply_draft  # Add reply draft to the row
        
        email_data.extend(rows)
        
        # Update status
        st.sidebar.text(f"Processing: {i+1}/{len(email_ids)}")
    
    progress_bar.empty()
    return email_data

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

def generate_reply_draft(sentiment, tone, explicit_requests, future_requests):
    """Generate a reply draft based on sentiment, tone, and requests."""
    reply = ""

    # Add tone-specific opening
    if tone == "Apologetic":
        reply += "We sincerely apologize for the inconvenience caused. "
    elif tone == "Grateful":
        reply += "Thank you for your email. We appreciate your feedback. "
    elif tone == "Urgent":
        reply += "Thank you for bringing this to our attention. We will address this immediately. "
    elif tone == "Frustrated":
        reply += "We understand your frustration and are working to resolve this issue. "
    else:
        reply += "Thank you for your email. "

    # Address explicit requests
    if explicit_requests:
        reply += "Regarding your request: "
        for req in explicit_requests:
            reply += f"{req} We will look into this and get back to you shortly. "

    # Address future requests
    if future_requests:
        reply += "For your future needs: "
        for req in future_requests:
            reply += f"{req} We will ensure this is handled in a timely manner. "

    # Add closing based on sentiment
    if sentiment in ["Highly Dissatisfied (1)", "Very Dissatisfied (2)", "Dissatisfied (3)"]:
        reply += "We value your feedback and are committed to improving our services. "
    elif sentiment in ["Highly Satisfied (10)", "Very Satisfied (9)", "Satisfied (8)"]:
        reply += "We are glad to hear your positive feedback and look forward to serving you again. "
    else:
        reply += "Please let us know if there is anything else we can assist you with. "

    reply += "\n\nBest regards,\n[Your Name]"
    return reply

# === STREAMLIT UI ===
def main():
    st.title("📊 Real-Time Sentiment Analysis on Emails")

    # Ensure session state for storing extracted emails
    if "email_data" not in st.session_state:
        st.session_state["email_data"] = None

    # Sidebar configuration
    with st.sidebar:
        st.header("🔑 Authentication")
        auth_tab, search_tab, advanced_tab = st.tabs(["Login", "Search", "Advanced"])

        with auth_tab:
            use_default = st.checkbox("Use default credentials", value=True)
            if use_default:
                email_user = config.DEFAULT_EMAIL
                email_pass = config.DEFAULT_PASSWORD
                st.success("Using saved credentials")
            else:
                email_user = st.text_input("Email", type="default")
                email_pass = st.text_input("Password", type="password")

        with search_tab:
            subject_query = st.text_input("Subject Keywords", "AA-1510")
            date_range = st.date_input(
                "Date Range",
                value=(
                    datetime.date.today() - datetime.timedelta(days=90),
                    datetime.date.today()
                )
            )

            # Handle single date selection
            if isinstance(date_range, datetime.date):
                date_since = date_range
            else:
                date_since = date_range[0]

            folder = st.selectbox(
                "Email Folder",
                ["INBOX", "Sent", "Drafts", "Archive", "Trash"],
                index=0
            )

        with advanced_tab:
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

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Extraction Controls")
        start_btn = st.button("🔍 Extract Emails", type="primary", use_container_width=True)

        if start_btn:
            if not email_user or not email_pass:
                st.error("Please enter email credentials!")
            else:
                with st.spinner("Connecting to email server..."):
                    mail = connect_to_email(email_user, email_pass)
                if mail:
                    with st.spinner(f"Searching for emails with subject '{subject_query}'..."):
                        email_ids = search_emails(
                            mail,
                            subject_query,
                            date_since.strftime("%d-%b-%Y"),
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

                        # Convert email data to DataFrame
                        df = pd.DataFrame(email_data)

                        # Sort emails by date and time to find the most recent one
                        try:
                            # Convert Date and Time to datetime for sorting
                            df["SortDate"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
                            df = df.sort_values(by=["SortDate"], ascending=False)
                            df = df.drop(columns=["SortDate"])
                        except Exception as e:
                            logger.warning(f"Error sorting by date: {e}")
                            # Fallback: sort by string comparison
                            df = df.sort_values(by=["Date", "Time"], ascending=False)

                        # Get the most recent email
                        most_recent_email = df.iloc[0]
                        reply_draft = most_recent_email["Reply Draft"]

                        # Save the DataFrame to session state
                        st.session_state["email_data"] = df

                        # Display extracted email data
                        st.write("### 📊 Extracted Email Data")
                        st.dataframe(
                            df,
                            height=400,
                            use_container_width=True
                        )

                        # Display the reply draft for the most recent email in a separate window
                        st.write("### ✉️ Reply Draft for the Most Recent Email")
                        with st.expander("View Reply Draft", expanded=True):  # Expanded by default
                            st.text_area(
                                "Reply Draft",
                                value=reply_draft,
                                height=300,
                                key="reply_draft"
                            )

                        # Download reply draft
                        st.download_button(
                            label="📥 Download Reply Draft",
                            data=reply_draft,
                            file_name="reply_draft.txt",
                            mime="text/plain",
                            key="download-reply-draft"
                        )

                    else:
                        st.warning("⚠️ No emails found matching your criteria")

                    # Close the connection
                    mail.logout()
                else:
                    st.error("Failed to connect to email server. Check credentials.")

    with col2:
        st.subheader("📋 Sentiment Analysis Controls")
        analyze_btn = st.button("🔍 Start Analysis")

        if analyze_btn:
            if "email_data" not in st.session_state or st.session_state["email_data"] is None or st.session_state["email_data"].empty:
                st.error("⚠️ No emails found! Extract emails first.")
            else:
                with st.spinner("Performing sentiment analysis..."):
                    email_df = st.session_state["email_data"].copy()  # Use stored DataFrame
                    email_df["Sentiment"] = email_df["Body"].apply(analyze_sentiment)

                st.success(f"✅ Processed {len(email_df)} emails!")
                st.dataframe(email_df, height=400, use_container_width=True)
                st.write("### 📊 Sentiment Distribution")
                st.bar_chart(email_df["Sentiment"].value_counts())

if __name__ == "__main__":
    main()