import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

class TweetClassifier:
    
    def __init__(self, **kwargs):
        """kwargs: 
        classifier (string, svm object)
        vectorizer (TfidfVectorizer object)
        """
        if kwargs['classifier']:
            self.classifier = self.load_classifier(kwargs['classifier'])
        else:
            self.classifier = svm.LinearSVC()
      

     def load_classifier(self, cl):
         """Assign classifier object or give the path to a pickled classifier"""
            if isinstance(cl, basestring):
                #load classifier
                classifier = pickle.load(open(cl, 'rb'))
            else:
                classifier = cl

        self.training_data = []
        self.labels = []
        self.labeled_data = []
        vectorizer = TfidfVectorizer(
                max_df = 0.8,
                sublinear_tf=True,
                use_idf=True)
        self.vectorizor = kwargs.get('vectorizer', vectorizer)

        self.vectors = None


    def add_labeled_data(self, data):
        """add a single labeled string or list of labeled strings"""
        if isinstance(data, tuple):
            self.labeled_data.append(data)
        elif isinstance(data, list):
            self.labeled_data.extend(data)
        else:
            raise TypeError('data must by a single tuple (text, label) or list of tuples')
    

    def add_data(self, data, label):
        """add single strings or list of strings specifying the label to attach
        to the points"""

        if isinstance(data, basestring):
            self.labeled_data.append((data, label))
        else:
            for s in data:
                self.labeled_data.append((s, label))
   
   def clear_data(self):
       """clear all data. Does not clear classifier"""
       self.training_data = []
       self.labeled_data = []
       self.labels = []
       self.vectors = None


    def prepare_data(self):
        """vectorize dataset"""
        if not self.labeled_data:
            warning.warn('No data to prepare!')
            return

        self.training_data = [x[0] for x in self.labeled_data]
        self.training_labels = [x[1] for x in self.labeled_data]
        self.vectors = vectorizer.fit_transform(self.training_data) 
    

    def train(self):
        """perform training on current data"""
        if not self.vectors:
            warnings.warn('Data not prepared!! Use prepare_data()!')
            return
        self.classifier.fit(self.training_data, self.training_labels)


    def predict(data):
        """classify single string or list of strings"""
        if isinstance(data, basestring)
            vectors = self.vectorizer.transform([data])   
        elif isinstance(data, list):
            vectors = self.vectorizer.transform(data)
        else:
            raise TypeError('data must be string or list of strings')
        
        return self.classifier.predict(vectors)
    

    def save_classifier(self, fname):
        pickle.dump(self.classifier, open(fname, 'wb'))
    

    def save_data(self, fname):
        pickle.dump(self.labeled_data, open(fname, 'wb'))




