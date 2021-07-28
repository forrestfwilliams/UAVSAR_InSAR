import os
import smtplib, ssl

def emailme(msg='Your script is complete!'):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "pymessage01@gmail.com"  # Enter your address
    receiver_email = "forrestfwilliams@icloud.com"  # Enter receiver address
    password = os.environ['PYPASSWORD']
    message = f'Subject: Message From Python\n\n{msg}\n\n<<Sent from Python>>'
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def main():
    msg = 'Your script is done!'
    emailme(msg)

if __name__ == '__main__':
    main()