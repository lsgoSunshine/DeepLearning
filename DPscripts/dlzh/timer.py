import time
class Timer(object):
    """Record multiple running times."""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # start the timer
        self.start_time = time.time()

    def stop(self):
        # stop a timer and record a time into a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)
