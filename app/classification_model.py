import pickle

class Model:
    def __init__(self) -> None:
        self.model = None
        self.vect = None
        self.load_model()
        self.load_vect()
        
    
    def load_model(self):
        self.model = pickle.load(open('model_3.pkl', 'rb'))
    
    def load_vect(self):
        self.vect = pickle.load(open('vect_model_3.pkl','rb'))
    
    def predict(self, data:list):
        processed_data = self.vect.transform(data)
        return self.model.predict(processed_data)