import spacy
import pickle
import string
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
lemma = WordNetLemmatizer()

class PrepData:
    def __init__(self, config:dict):
        self.spacy_n_process = config.get('spacy_n_process')
        self.spacy_batch_size = config.get('spacy_bath_size')
        self.oversample_strategy = config.get('oversample_strategy')
        self.undersample_strategy = config.get('undersample_strategy')
        self.train_size = config.get('train_size')
        self.data_chunk = config.get('data_chunk')
        self.vect_pickle_path = config.get('vect_pickle_path')
        self.model_pickle_path = config.get('model_pickle_path')
        self.path_dataset = config.get('path_dataset')
    
    def open_dataset(self):
        data = pd.read_csv(self.path_dataset, iterator=True)
        data = data.get_chunk(self.data_chunk)
        return data.text, data.label
        
    def remove_punct(self, X:list):
        clean_punct_text = [str(review).lower().translate(str.maketrans('', '', string.punctuation)) for review in X]
        return clean_punct_text
    
    def remove_stopwords(self, X:list):
        stopwords = nlp.Defaults.stop_words

        clean_no_stopwords = []
        for review in X:
            review = set(review.split())
            clean_review = review.difference(stopwords)
            clean_no_stopwords.append(' '.join(clean_review))
        
        return clean_no_stopwords
    
    def lemmatize(self, X:list):
        lemmatize_data = []
        
        for review in X:
          review_split = review.split()
          lemma_words = [lemma.lemmatize(i) for i in review_split]
          lemmatize_data.append(' '.join(lemma_words))
        
        return lemmatize_data
    
    def tfid_vectorize(self, X:list):
        tfid_vect = TfidfVectorizer()
        vectorize_data = tfid_vect.fit_transform(X)
        pickle.dump(tfid_vect, open(self.vect_pickle_path,'wb'))
        return vectorize_data
    
    def split_data(self, X:list, y:list):
        return train_test_split(X, y, train_size=self.train_size, shuffle=True)
    
    def under_over_sample(self, X:list, y:list):
        ros = RandomOverSampler(sampling_strategy=self.oversample_strategy)
        rus = RandomUnderSampler(sampling_strategy=self.undersample_strategy)
        X_ros, y_ros = ros.fit_resample(X, y)
        return rus.fit_resample(X_ros, y_ros)
    
    def fit_transform(self):
        X,y = self.open_dataset()
        no_punct = self.remove_punct(X)
        no_stopwords = self.remove_stopwords(no_punct)
        lemmatized_data = self.lemmatize(no_stopwords)
        vectorize_data = self.tfid_vectorize(lemmatized_data)
        X_train, X_test, y_train, y_test = self.split_data(vectorize_data, y)
        X, y = self.under_over_sample(X_train, y_train)
        return X, y, X_test, y_test

class Model(PrepData):
    def __init__(self, config: dict, model_pickle_path):
        super().__init__(config)
        self.X, self.y, self.X_test, self.y_test = self.fit_transform()
        self.trained_model = None
        self.model_pickle_path = model_pickle_path
    
    def train(self):
        model = ComplementNB()
        model.fit(self.X, self.y)
        self.trained_model = model
        pickle.dump(model,open(self.model_pickle_path,'wb'))
        
    def predict(self):
        return self.trained_model.predict(self.X_test)
    
    def benchmark(self):
        y_pred = self.predict()
        print(f1_score(self.y_test, y_pred))
        print(accuracy_score(self.y_test, y_pred))
        print(precision_score(self.y_test, y_pred))
        print(recall_score(self.y_test, y_pred))

config = {'spacy_n_process':-1,
          'spacy_bath_size':100,
          'oversample_strategy':.8,
          'undersample_strategy':.9,
          'train_size':.9,
          'data_chunk':3000000,
          'vect_pickle_path':'/content/drive/MyDrive/data/yelp_review/vect_model.pkl',
          'path_dataset': '/content/drive/MyDrive/data/yelp_review/binary_data.csv',}

model = Model(config=config,model_pickle_path='/content/drive/MyDrive/data/yelp_review/model.pkl')
model.train()
model.benchmark()