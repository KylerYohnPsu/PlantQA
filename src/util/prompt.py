from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class Prompt:
    question: str
    image: np.array
    storing_permission_granted: bool
    job_id: any
    creation_timestamp: float = field(init=False)

    def __str__(self):
        return (
            f"Question: {self.question}\n"
            f"Permission granted: {self.storing_permission_granted}\n"
            f"Job ID: {self.job_id}"
        )


    def __post_init__(self):
        self.creation_timestamp= time.time()