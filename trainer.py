from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

def teacher(labeled_data):
    """takes list of tuples of tweet text string and
    returns trained support vector classifier"""
    
    training_data = [x[0] for x in labeled_data]
    training_labels = [x[1] for x in labeled_data]
    #transform into feature vectors
    vectorizer = TfidfVectorizer(
            max_df = 0.8,
            sublinear_tf=True,
            use_idf=True)
    training_vectors = vectorizer.fit_transform(training_data) 
    #linear SVC
    classifier = svm.LinearSVC()
    classifier.fit(training_data, training_labels)
    return classifier




