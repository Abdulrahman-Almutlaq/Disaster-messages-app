import sys
import pandas as pd
from sqlalchemy.engine import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import nltk
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import numpy as np
import joblib


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(database_filepath):
    """
    Loads data from the database into a pandas dataframe.

    Param database_filepath: Dataframe object of the two merged datasets.

    Returns the features of the training data X.
    Returns the labels of the training data Y.
    """
    
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('Cleaned_disaster_data',engine)
    non_target = ['id_messages', 'original', 'genre', 'message']
    X = df.message
    Y = df[(df.columns[~df.columns.isin(non_target)])]

    return X,Y


def tokenize(text): 
    """
    Processing and tokenizing the passed text

    Param text: the raw text of messages.

    Returns the tokens.

    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokenized = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(w) for w in tokenized if w not in stopwords.words("english")]
    
    tokens = [lemmatizer.lemmatize(w, pos='v') for w in tokens]
    
    return tokens


def build_model():
    """
    Setting up the pipeline for the model

    Returns GridSearchCV object to be fitted.
    """
    
    pipeline = Pipeline([('text_pipeline',Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                    ])),
                        ('MultiOutputClassifier', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = [{
    'MultiOutputClassifier__estimator__n_estimators' : [10, 100]
}]

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

    


# Param category_names: the notebook implementation doesn't require one so it was deleted
def evaluate_model(model, X_test, Y_test):
    """
    Prints a classification_report for the model passed.

    Param model: the model to evaluate.
    Param X_test: the features of the testing data X.
    Param Y_test: the labels of the testing data Y.
    """
    y_pred = model.predict(X_test)

    for ind, col in enumerate(Y_test):
        
        print(col, ':\n', classification_report(Y_test[:, ind], y_pred[:, ind]),"\n\n")




def save_model(model, model_filepath):
    """
    Exports the model passed (in a pickle format).

    Param model: the model to save.
    Param model_filepath: the path of the model to be saved.
    """
    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()