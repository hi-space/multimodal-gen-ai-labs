import secrets
from datetime import datetime


def get_current_time():
    return datetime.now().strftime('%y-%m-%d %H:%M:%S')

def get_seed():
    return secrets.randbelow(2147483647)