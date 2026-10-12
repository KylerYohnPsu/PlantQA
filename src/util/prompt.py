from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class Prompt:
    question: str
    image: np.array
    job_id: any
    creation_timestamp: float = field(init=False)

    def __post_init__(self):
        self.creation_timestamp= time.time()