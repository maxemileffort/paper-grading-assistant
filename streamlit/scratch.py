from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

df = pd.read_csv('./sample_data/processed_essays.csv')

df = df.loc[df.word_count > 25].drop(columns=['Unnamed: 0'])

print(df.head())

tf_vectorizer = CountVectorizer(max_df=0.85, 
                                min_df=1, 
                                max_features=1000, 
                                stop_words=None,) 
                                # preprocessor=' '.join)
tf = tf_vectorizer.fit_transform(df['tokenized_essay'])

X = tf
y = df['class']

clf = SVC(kernel = 'rbf', random_state = 0)

clf.fit(X, y)

dump(clf, './models/kaggle_trained_clf_model.joblib') 
dump(tf_vectorizer, './models/kaggle_trained_tf_model.joblib') 