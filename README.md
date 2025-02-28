# 📨 Email Sentiment Analysis Tool

This tool allows Journal Managers to extract emails, analyze their sentiment, and generate reply drafts for communication with authors and publishers. It uses Streamlit for the user interface and a pre-trained sentiment analysis model for tone and sentiment classification.


## 🚀 Features

- **Email Extraction**: Connect to an IMAP server and extract emails based on subject, date range, and folder.
- **Sentiment Analysis**: Analyze the sentiment and tone of each email.
- **Reply Draft Generation**: Automatically generate reply drafts for the most recent email.
- **User-Friendly UI**: Built with Streamlit for an intuitive and interactive experience.
- **Downloadable Reports**: Download extracted email data and reply drafts as CSV and text files.


## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/email-sentiment-analysis.git
   cd email-sentiment-analysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Email Credentials**:
   - Create a `config.py` file in the project directory.
   - Add your email credentials and IMAP server details:
     ```python
     IMAP_SERVER = "imap.example.com"  # Replace with your IMAP server
     DEFAULT_EMAIL = "your-email@example.com"  # Replace with your email
     DEFAULT_PASSWORD = "your-password"  # Replace with your password
     ```

## 🖥️ Usage

1. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

2. **Access the UI**:
   - Open your browser and go to `http://localhost:8501`.
   - Use the sidebar to authenticate and configure email search criteria.
   - Click "Extract Emails" to process emails and generate reply drafts.

3. **View Results**:
   - Extracted email data is displayed in a table.
   - The reply draft for the most recent email is shown in a separate window.
   - Sentiment analysis results are displayed in a bar chart.

4. **Download Reports**:
   - Download the extracted email data as a CSV file.
   - Download the reply draft as a text file.

## 📂 Project Structure

```
email-sentiment-analysis/
├── main.py       # Main script for email extraction and analysis
├── config.py              # Configuration file for email credentials
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.


## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing UI framework.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the pre-trained sentiment analysis model.
- [IMAPClient](https://imapclient.readthedocs.io/) for IMAP email handling.