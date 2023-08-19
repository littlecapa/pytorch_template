import logging
import json

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
    
class Summary():
    def __init__(self):
        self.summary = "Summary:" + "\n"

    def add(self, summary, stage = "Training"):
        self.add_stage(stage)
        self.summary += json.dumps(summary) + "\n"

    def add_stage(self, stage):
        self.summary += f"Stage: {stage}"
    
    def get_all(self):
        return self.summary