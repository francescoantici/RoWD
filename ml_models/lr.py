from sklearn.linear_model import LogisticRegression

class LR:
    
    name = "LR"
    
    def __init__(self, **kwargs):
        self._model =  LogisticRegression(**kwargs)
    

    