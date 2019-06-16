import json
import datetime as dt
import numpy as np


class jsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dt.datetime):
            return obj.strftime("%Y/%m/%d %H:%M:%S")
        elif isinstance(obj, dt.timedelta):
            return str(obj)
        else:
            return super().default(obj)


class Timer:
    def __init__(self, name=None):
        self.name = "'" + name + "'" if name else ""

    def __enter__(self):
        self.start = dt.datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = dt.datetime.now()
        self.time = self.end - self.start

    def __repr__(self):
        return f"{self.name} = {self.time}"

