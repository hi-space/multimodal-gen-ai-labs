from datetime import datetime


def get_current_time():
    return datetime.now().strftime('%y-%m-%d %H:%M:%S')
