import time
import numpy as np
import os 

class Timer:
    timers = dict()

    def __init__(self, log = None, timer = False):
        self._start_time = None
        self.log = log
        self.timer = timer

    def start(self):
        if self.timer:
            self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and saves the elapsed time"""
        if self.timer:
            if self._start_time is None:
                raise ValueError('stopped timer without starting it')

            elapsed_time = time.perf_counter() - self._start_time
            self._start_time = None

            if not self.log in self.timers:
                self.timers[self.log] = []
            
            self.timers[self.log].append(elapsed_time)
    
    def __enter__(self):
        """Start a new timer as a context manager"""
        if self.timer:
            assert self.log, "must specify what to log if using Timer() as a context manager"
            self.start()
            return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        if self.timer:
            self.stop()

    def summary(self):
        if self.timer:
            self.means  = dict()
            self.stds = dict()
            self.counts = dict()
            for key, value in self.timers.items():
                self.means[key] = np.mean(value)
                self.stds[key] = np.std(value)
                self.counts[key] = len(value)

    def save(self, savedir):
        if self.timer:
            self.summary()
            with open(savedir, "w") as fp:
                for key in self.timers:
                    fp.write("{}: {:.5f} +/- {:.5f}. counts: {}\n".format(key, self.means[key], self.stds[key], self.counts[key]))
