import pandas as pd
import numpy as np
import nltk
import re
import unidecode
##########################################################################
print("import keras...")
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.models import Model, load_model
from keras.utils import np_utils, to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten, GRU, Bidirectional
from keras.layers import Embedding, SpatialDropout1D, Activation, Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
##########################################################################
print("import sklearn...")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, metrics
##########################################################################
NClean=0
##########################################################################
def Clean_Text(Text):
    global NClean   
    
    #verificar regex!!!!
    
    Text = Text.lower() # lowercase
    StopML="de|en|em|para|con|i|sin|a|y|al|la|por|el|com|do|by|promo|envio|producao|cm|mm|oferta|producto|cuotas|interes|oportunidad"
    REPLACE_STOP = re.compile("\\b("+StopML+")\\b")    
    LEAVE_ONLYCHARS = re.compile('[^a-z ]')
    REPLACE_1LETTER = re.compile("\\b[a-z ]\\b")
    REPLACE_2LETTER = re.compile("\\b[a-z]{2}\\b")
    REPLACE_BAD_SPACE = re.compile('\s+')

    Text = unidecode.unidecode(Text)
    Text = LEAVE_ONLYCHARS.sub(' ', Text)
    Text = REPLACE_1LETTER.sub(' ', Text) 
    Text = REPLACE_2LETTER.sub(' ', Text) 
    Text = REPLACE_STOP.sub(" ", Text)
    Text = REPLACE_BAD_SPACE.sub(' ', Text)
    
    NClean+=1
    if(NClean%10000)==0:
        print("N=%u  values"%NClean, end="\r")

    return Text
##########################################################################   
##########################################################################

Train_Data = pd.read_csv('data/train.csv', nrows=200000 )
print("Elementos de training = %u" % ( len(Train_Data) ) )
Test_Data = pd.read_csv('data/test.csv', nrows=200000 )
print("Elementos de testing = %u" % ( len(Test_Data) ) )

NClean=0
print("Limpiando dataset train...")
Train_Data['title'] = Train_Data['title'].apply(Clean_Text)

LEncoder = LabelEncoder()
LEncoder.fit(Train_Data['category'])
Train_Y_Le = LEncoder.transform(Train_Data['category'])

Train_TfIdf = Train_Data['title'].values.tolist()
Test_TfIdf = Test_Data['title'].values.tolist()

Nw=Train_Data['title'].apply(lambda x: len(x.split(' '))).sum()
print("Nw=",Nw)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(Train_TfIdf, Train_Y_Le, test_size=0.3, random_state = 42)

NBClassifier = Pipeline( [ ('vect', CountVectorizer(ngram_range=(2,2))), ('tfidf', TfidfTransformer(ngram_range=(2,2))), ('clf', MultinomialNB()) ])

NBClassifier.fit(X_Train, Y_Train)

Y_Pred = NBClassifier.predict(X_Test)

print('accuracy %s' % accuracy_score(Y_Pred, Y_Test))





