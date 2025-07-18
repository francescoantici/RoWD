from typing import List, Iterable
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from feature_encoders.feature_encoder import FeatureEncoder

class SBEncoding(FeatureEncoder):
        
    def __init__(self, weights = "all-MiniLM-L6-v2") -> None:
       self._encoder = SentenceTransformer(weights)
    
    def encode_data(self, data: Iterable) -> List:
        """
        Encodes the data.
        
        :param data: The job scripts.
        :return: List of encoded features.
        """
        # Encoding the job data
        encodings = np.zeros((len(data), 384))
        for i in range(len(data)):
            encodings[i] = self.encode_job(data[i])
        return encodings
       
    def encode_job(self, job_script:str) -> Iterable:
        """
        Encodes the job script.
        
        :param x: The job script.
        :return: List of encoded features.
        """
        try:
            return self._encoder.encode(self._parse_job_data(job_script)) 
        except Exception as e:
            print(e)
            return []
