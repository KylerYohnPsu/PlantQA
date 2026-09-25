'''
Steps to Agent

1. Recieve question and Visual breakdown (either class or separated by features of a plant)

2. preprocess features of plant and get returned 
'''

"""Are we able to leverage the local host model"""
"""Add in crowd sourced data to the model"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from src.util.logger import Logger
from src.data_preprocessing.TextEmbedder import TextEmbedder, EmbeddingRecord


