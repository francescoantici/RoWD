from sklearn.ensemble import RandomForestClassifier

class RF:
    
    name = "RF"
    
    def __init__(self, **kwargs):
        self._model =  RandomForestClassifier(**kwargs)
    
