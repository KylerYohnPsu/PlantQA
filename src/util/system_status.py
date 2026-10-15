from enum import Enum

class SystemStatus(Enum):
    READY= "ready"
    PREPROCESSING= "preprocessing"
    PROCESSING= "processing"
    PREDICTING= "predicting"
    GENERATING= "generating"
    EMBEDDING= "embedding"
    ERROR= "error"
    DISCONNECTED= "disconnected"
    INITIALIZING= "initializing"