##########################################################################
# Cristian Belussi. 2019.
# github.com/xtianhb/MeLiCha2019 
# Licencia MIT.
# MeLi Data Challenge.
##########################################################################
#ToDo: Mejorar codigo y comentar.
##########################################################################
####
DoPre=0  #Leer, limpiar, y guardar datos
PreReadTrain=0 #Train
PreReadTest=0 #Test
CleanTrain=0 #Clean
CleanTest=0 #Clean
####
DoModel=1 
ReadTrain=0 #Leer csv
DoCats=0 #Proc cats csv y guardar
DoToken=0 #Gen tokens csv y guardar
DoSeqPadSplit=0 #Split datos y guardar, sino lee de disco

DoTrainModel=1 #Entrenar
DoImportChkTrain=1  #Importar
#Ensemble
EvaluateOnTrain=0
DoEvaluate=1 #Evaluar
DoTestData=1 #Test completo
DoSubmit=1 #MeLi

NN=1
EPOCHS=1
NROWS=2000000
CHUNKSIZE=250000
MAX_SEQ_LEN = 8
BATCH_SIZE = 1024
NClean=0
ScoreMeLi=0.8318
##########################################################################
#Python
import os
import re
import pickle
##########################################################################
#Numpy & Pandas
import pandas as pd
import numpy as np
import unidecode
##########################################################################
#Sci Kit Learn
print("Import sklearn...")
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
##########################################################################
print("import keras...")
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.models import Model, load_model
from keras.utils import np_utils, to_categorical
from keras.layers import Concatenate, Conv1D, MaxPool1D, Average
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten, GRU, Bidirectional
from keras.layers import Embedding, SpatialDropout1D, Activation, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
##########################################################################
def Clean_Text(Text):
    global NClean   
    
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
def PreProcess():
    #Data reading
    
    global NClean
    
    if PreReadTrain:
        # RawTrain: Index(['title', 'label_quality', 'language', 'category'], dtype='object')
        print("Leyendo dataset train...")
        RawTrain = pd.read_csv('data/train.csv', nrows=NROWS)
        print("Elementos de training = %u" % ( len(RawTrain) ) )
        print("TRAIN ANTES DE LIMPIAR:")
        print(RawTrain.columns)
        print(RawTrain[0:10])
        ##########################################
        #Text processing. Data Clean & Pre-Process
        Train_Data = RawTrain
        if CleanTrain:
            print("Limpiando dataset train...")
            NClean=0
            Train_Data['title'] = Train_Data['title'].apply(Clean_Text)
        print("Elementos de training limpios = %u " % ( len(Train_Data) ) )
        print("TRAIN DESPUES DE LIMPIAR:")
        print(RawTrain.columns)
        print(Train_Data[0:10])
        print("Guarda copia de train clean")
        Train_Data.to_csv('data/trainclean.csv', index=False, na_rep=" ")
    
    if PreReadTest==1:
        # RawTest: Index(['id', 'title', 'language'], dtype='object')
        print("Leyendo dataset test...")
        RawTest = pd.read_csv('data/test.csv')
        print("Elementos de testing = %u" % ( len(RawTest) ) )
        print(RawTest.columns)
        print("TEST ANTES DE LIMPIAR:")
        print(RawTest[0:10])
        Test_Data = RawTest
        if CleanTest:
            print("Limpiando dataset test...")
            NClean=0
            Test_Data['title'] = Test_Data['title'].apply(Clean_Text)
        print("Elementos de testing limpios = %u" % ( len(RawTest) ) )
        print("TEST DESPUES DE LIMPIAR:")
        print(RawTest.columns)
        print(Test_Data[0:10])
        print("Guarda copia de test clean")
        Test_Data.to_csv('data/testclean.csv', index=False, na_rep=" ")
    
    return
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
def ModelNLP():
    global ScoreMeLi
    
    print("ScoreMeLi=",ScoreMeLi)
    if EvaluateOnTrain:
        print("ReadTrain=",ReadTrain)
        #Data
        print("Reading dataset train...")
        Start = np.random.randint(1, NROWS-CHUNKSIZE)
        print("Lee Train_Data desde %u a %u"%(Start, Start+CHUNKSIZE))
        Train_Data = pd.read_csv('data/trainclean.csv', skiprows = range(1, Start), nrows=CHUNKSIZE)
        Train_Data.dropna(inplace=True)
        print("Elementos de training = %u" % ( len(Train_Data) ) )

    #Categories
    with open('DictCats.pkl', 'rb') as Handle:
        DictCats = pickle.load(Handle)
    with open('DictKeys.pkl', 'rb') as Handle:
        DictKeys = pickle.load(Handle)
    with open('Labels.pkl', 'rb') as Handle:
        Labels = pickle.load(Handle)
    
    #Tokens
    print("DoToken=",DoToken)
    with open('Tokens.pkl', 'rb') as Handle:
        DataTokenizer = pickle.load(Handle)

    Vocab_Size = len(DataTokenizer.word_index) + 1
    print('Hay %s tokens.' % Vocab_Size)
    
    if EvaluateOnTrain:
        print("category replace")
        Train_Data["category"].replace(DictKeys, inplace=True)
        print("To categorical")
        Y_Enc = to_categorical(Train_Data['category'].values)
        print("Data to seq...")
        Train_Data_Seq = DataTokenizer.texts_to_sequences(Train_Data.title.values)
        print("Seq padding...")
        Train_Data_Pad = pad_sequences(Train_Data_Seq, maxlen=MAX_SEQ_LEN)
        print("Splitting dataset...")
        XTrain, XTest, YTrain, YTest = train_test_split(Train_Data_Pad, Y_Enc, test_size=0.1, random_state = 42, shuffle=True)                            
    
    NLABELS = len ( Labels )
    print("Hay %u labels " % NLABELS)

    print("NN=",NN)
    #Neural Network Train / Test
    
    #NN==1:#NOTOCAR
    ModelT1 = Sequential() #0,80 OK 15 mins / epoch 2M
    ModelT1.add(Embedding(input_dim=Vocab_Size, output_dim=128, input_length=MAX_SEQ_LEN) ) 
    ModelT1.add(Flatten())
    ModelT1.add(Dense(256, activation='relu'))
    ModelT1.add(Dense(NLABELS, activation='softmax') )
    FilePathWeights1="Weights_NN1.hdf5"
    ModelT1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    ModelT1.summary()
    ModelT1.load_weights(FilePathWeights1)
    
    #NN==2:#OPTIMIZAR
    # ModelT2 = Sequential()  #0,81
    # ModelT2.add(Embedding(input_dim=Vocab_Size, output_dim=512, input_length=MAX_SEQ_LEN) ) 
    # ModelT2.add(Flatten())
    # ModelT2.add(Dense(2048, activation='relu'))
    # ModelT2.add(Dropout(0.5))
    # ModelT2.add(Dense(NLABELS, activation='softmax') )
    # FilePathWeights2="Weights_NN3.hdf5"
    # ModelT2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # ModelT2.summary()
    # ModelT2.load_weights(FilePathWeights2)
    
    #NN==3:#NOTOCAR
    ModelT3 = Sequential()  #0,81
    ModelT3.add(Embedding(input_dim=Vocab_Size, output_dim=512, input_length=MAX_SEQ_LEN) ) 
    ModelT3.add(Flatten())
    ModelT3.add(Dense(2048, activation='relu'))
    ModelT3.add(Dropout(0.5))
    ModelT3.add(Dense(NLABELS, activation='softmax') )
    FilePathWeights3="Weights_NN3.hdf5"
    ModelT3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    ModelT3.summary()
    ModelT3.load_weights(FilePathWeights3)
    Models = [ModelT1, ModelT3]
    
    BAcc=0
    if EvaluateOnTrain:
        print("Evaluate model Acc...")
        Acc1=ModelT1.evaluate(XTest, YTest)
        print('##########Test set\n  Loss: {:0.3f}  Accuracy: {:0.3f}##########'.format(Acc1[0], Acc1[1]))
        Acc3=ModelT3.evaluate(XTest, YTest)
        print('##########Test set\n  Loss: {:0.3f}  Accuracy: {:0.3f}##########'.format(Acc3[0], Acc3[1]))
        
        print("Evaluate model B-Acc...")
        YPred1 = ModelT1.predict(XTest, verbose=1)
        YPred3 = ModelT3.predict(XTest, verbose=1)
        YPred = (YPred1+YPred3)/2.0
        YPredMax = np.argmax(YPred, axis=1).tolist()
        YTestMax = np.argmax(YTest, axis=1).tolist()

        BAcc = balanced_accuracy_score(YTestMax, YPredMax)
        print ("########## B-Acc: %0.3f ##########" % BAcc )

    if (ScoreMeLi>BAcc) or (notEvaluateOnTrain):
        print("New score = ", ScoreMeLi)
        print("Read test data")
        Test_Data = pd.read_csv('data/testclean.csv')
        Test_Data = Test_Data.fillna(' ')
        print("Elementos de test clean = %u " % ( len(Test_Data) ) )
        
        print("Data to sequences...")
        Test_Data_Seq = DataTokenizer.texts_to_sequences(Test_Data.title.values)
        print("Sequences padding...")
        Test_Data_Pad = pad_sequences(Test_Data_Seq, maxlen=MAX_SEQ_LEN)

        print("Test predict...")
        Test_Res_Prob1 = ModelT1.predict(Test_Data_Pad, verbose=1, batch_size=BATCH_SIZE)
        Test_Res_Prob3 = ModelT3.predict(Test_Data_Pad, verbose=1, batch_size=BATCH_SIZE)
        print("Test arg max...")
        Test_Res_Prob = (Test_Res_Prob1+Test_Res_Prob3)/2.0
        Test_Res_Int = np.argmax(Test_Res_Prob, axis=1).tolist()
        print("Fin!")
       
        Test_Res =  [ DictCats[C] for C in Test_Res_Int ] 
        print("Elementos de Test_Res = %u " % ( len(Test_Res) ) )
    
        print("DoSubmit=", DoSubmit)
        Test_Data.insert(3, "category", Test_Res)
        Test_Data['category'] = Test_Data['category'].str.upper() 
        _S = np.random.randint(low=0, high=10000)
        print( Test_Data[ _S:_S+10 ] )
        Submit = Test_Data[["id","category"]]
        Submit.to_csv('data/xtian_ens.csv', index=False)
    else:
        print("No se hace procesa submit porque ScoreMeLi(%f)<Acc(%f)"%(ScoreMeLi, Acc_A))
        
        
    return
##########################################################################
if DoPre:
    PreProcess()

if DoModel:
    ModelNLP()

##########################################################################

