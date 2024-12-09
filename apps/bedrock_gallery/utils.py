
from datetime import datetime


def format_datetime(time_str: str, seconds=False) -> str:
    dt = datetime.fromisoformat(time_str)
    if seconds:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return dt.strftime("%Y년 %m월 %d일 %H시 %M분")
        