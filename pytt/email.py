import email, smtplib, ssl
from getpass import getpass
import torch.distributed as dist
from threading import Thread
from pytt.logger import logger

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def default_onfinish():
    logger.log("Done sending email")

def default_onerror(e):
    logger.log("Error sending email!!!")
    raise e

def check_attachment_error(e):
    return isinstance(e, smtplib.SMTPSenderRefused)\
           and e.smtp_code == 552

def get_email_info(smtp_server=None, port=None, sender_email=None, sender_password=None, receiver_email=None):
    ss = input("SMTP Server: ") if smtp_server is None else smtp_server
    p = input("Port on %s: " % ss) if port is None else port
    se = input("Sender email (for %s): " % ss) if sender_email is None else sender_email
    sp = getpass("Sender password for %s: " % se) if sender_password is None else sender_password
    re = input("Receiver email: ") if receiver_email is None else receiver_email
    return ss, p, se, sp, re

class EmailSender:
    def __init__(self, smtp_server=None, port=None, sender_email=None, sender_password=None, receiver_email=None, subject="Process Notification from pytt"):
        if dist.is_initialized() and (smtp_server is None or
                                      port is None or
                                      sender_email is None or
                                      sender_password is None or
                                      receiver_email is None):
            raise Exception
        self.smtp_server, self.port, self.sender_email, self.password, self.receiver_email = get_email_info(
            smtp_server=smtp_server, port=port, sender_email=sender_email, sender_password=sender_password, receiver_email=receiver_email)
        self.subject = subject

    # attachments have the form of a generator of (name, filename, file) tuples
    def __call__(self, body, attachments=[], onfinish=default_onfinish, onerror=default_onerror):
        thread = Thread(target=self.send_email, args=[body, attachments, onfinish, onerror])
        thread.start()

    def send_email(self, body, attachments=[], onfinish=default_onfinish, onerror=default_onerror):
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = self.subject
        #message["Bcc"] = receiver_email  # Recommended for mass emails

        # Add body to email
        message.attach(MIMEText(body, "plain"))

        # add attachments
        for name,filename,file in attachments:
            self.add_attachment(message, name, filename, file)

        # send email
        text = message.as_string()
        try:
            send_email(self.smtp_server, self.port, self.sender_email, self.password, self.receiver_email, text)
            onfinish()
        except Exception as e:
            onerror(e)

    def add_attachment(self, message, name, filename, file):
        part = MIMEBase("application", "octet-stream")
        part.set_payload(file.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            'Content-Disposition', 'attachment', filename=filename,
        )

        # Add attachment to message
        message.attach(part)

def send_email(smtp_server, port, sender_email, password, receiver_email, text):
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

