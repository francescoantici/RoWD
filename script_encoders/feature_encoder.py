import pandas as pd 
from typing import List, Iterable

class FeatureEncoder:
    """
    Base class for feature encoders.
    """

    def encode_data(self, data: Iterable) -> List:
        """
        Encodes the data.
        
        :param data: The job scripts.
        :return: List of encoded features.
        """
        pass
    
    def encode_job(self, job_script: str) -> list:
        """
        Encodes the job script.
        
        :param x: The job script.
        :return: List of encoded features.
        """
        pass
    