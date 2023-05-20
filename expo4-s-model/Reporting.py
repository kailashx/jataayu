import smtplib
import ssl
import stomp

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "chnadra97762@gmail.com"  # Enter your address
receiver_email = "zoook.research@gmail.com"  # Enter receiver address
password = "njzkavmbmshwcxpu"
message = """\
Subject: Home Security Alert!!!

Hey Kailasha,
    
    Unusual activity is detected in your home, please respond quickly.

    Regards,
    Jataayu
"""

# Create a secure SSL context
context = ssl.create_default_context()


def send_email():
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        # TODO: Send email here
        server.sendmail(sender_email, receiver_email, message)

'''
# Create a Listener class inheriting the stomp.ConnectionListener class
# stomp.ConnectionListener class definition can be found here:
# https://github.com/jasonrbriggs/stomp.py/blob/2435108cfc3eb4bd6477653b071e85acd6a2f211/stomp/listener.py
class Listener(stomp.ConnectionListener):
    # Override the methods on_error and on_message provides by the
    # parent class
    def on_error(self, frame):
        print('received an error "%s"' % frame)

        # Print out the message received

    def on_message(self, frame):
        print('received a message "%s"' % frame)

    def on_send(self, frame):
        print('sent a message "%s"' % frame)


# Declare hosts as an array of tuples containing the ActiveMQ server # IP address or hostname and the port number
hosts = [('localhost', 61613)]

# Create a connection object by passing the hosts as an argument
conn = stomp.Connection(host_and_ports=hosts)

# Tell the connection object to listen for messages using the
# Listener class we created above
conn.set_listener('', Listener())

# Initiate the connection with the credentials of the ActiveMQ server
conn.connect('admin', 'admin', wait=True)

# Register a consumer with ActiveMQ. This tells ActiveMQ to send all # messages received on the queue 'queue-1' to
# this listener
conn.subscribe(destination='/queue/shutdown', id='2', ack='auto')
'''