import imaplib
import ssl

# IMAP Server Details
IMAP_SERVER = "imap.gmail.com"  # Replace with your IMAP server
IMAP_PORT = 993  # Standard IMAP over SSL port
EMAIL = "srikanth.mohan@tnqtech.com"  # Replace with your email
PASSWORD = "vzgdbviniemxrvio"  # Replace with your password

try:
    # Create SSL context with strict verification to force SSL errors
    context = ssl.create_default_context()
    context.check_hostname = False  # Enforce hostname verification
    context.verify_mode = ssl.CERT_NONE  # Require valid certificate

    # Connect to IMAP server using SSL
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT, ssl_context=context)
    mail.login(EMAIL, PASSWORD)

    # Fetch mailboxes to confirm connection
    status, mailboxes = mail.list()
    if status == "OK":
        print("Mailboxes:", mailboxes)

    mail.logout()

except ssl.SSLCertVerificationError as ssl_err:
    print(f"SSL Certificate Verification Failed: {ssl_err}")
except ssl.SSLError as ssl_err:
    print(f"SSL Error: {ssl_err}")
except Exception as e:
    print(f"Error: {e}")