"""
This script runs the dsaas application using a development server.
"""

from os import environ
from dsaas import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
#https://www.youtube.com/watch?v=GgPr69uMq_g
#INTRA-DAY STOP LOSS HUNTING EXPLAINED - TAMIL