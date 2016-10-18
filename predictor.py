from sklearn.feature_extraction.text import TfidfVectorizer

class Predictor:
   """Simple class for predicting label of data.
    initialised with trained classifier"""
    def __init__(self, classifier):
        self.classifier = classifier

    def predict(data):
        if isinstance(data, basestring)
            vectors = vectorizer.transform([data])   
        elif isinstance(data, list):
            vectors = vectorizer.transform(data)
        else:
            raise TypeError('data must be string or list of strings')
        
        return classifier.predict(vectors)

        
