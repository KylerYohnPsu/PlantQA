from enum import Enum

class SystemStatus(Enum):
    READY= "ready"
    PREPROCESSING= "preprocessing"
    PROCESSING= "processing"
    ERROR= "error"
    DISCONNECTED= "disconnected"
    INITIALIZING= "initializing"