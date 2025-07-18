import pickle

class MLModel:
    
    name = "MlModel"
    
    def __init__(self, **kwargs):
        self._model =  None
    
    def train(self, x: list, y: list) -> bool:
        try:
            self._model = self._model.fit(x, y)
        except Exception as e:
            print(e)
            return False 
        else:
            return True
    
    def predict(self, x: list) -> list:
        try:
            return self._model.predict(x)
        except Exception as e:
            print(e)
            return []
    
    def save(self, path: str) -> bool:
        try:
            with open(path, 'wb') as f:
                pickle.dump(self._model, f)
        except Exception as e:
            print(e)
            return False
        else:
            return True
    
    def load(self, path: str) -> bool:
        try:
            with open(path, 'rb') as f:
                self._model = pickle.load(f)
        except FileNotFoundError:
            print(f"Model file not found at {path}. Please check the path and try again.")
            return False
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return False
        else:
            return True
    
    @property
    def model(self):
        return self._model