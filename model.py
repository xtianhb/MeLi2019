##########################################################################
# Cristian Belussi. 2019. 
# MeLi Challenge
######LAST SCORE  #####
##########################################################################
DoPreProc=0  #Leer, limpiar, y guardar datos
PreReadTrain=0 #Train
PreReadTest=0 #Test
CleanTrain=0 #Clean
CleanTest=0 #Clean
#################################
DoModel=1
ReadTrain=1 #Leer csv
DoCats=0 #Proc cats csv y guardar. NO NO NO.
DoToken=0 #Gen tokens csv y guardar. NO NO NO.
GetSeqPadSplit=1 #Split datos y guardar, sino lee de disco. Si.
#
DoTrainModel=1 #Entrenar
DoImportChkTrain=1     #Importar
DoEvaluate=1 #Evaluar
DoTestData=1 #Test completo
DoSubmit=1 #MeLi
##########################################################################
NN=2
NROWS=20000000
CHUNKSIZE=1500000
MAX_SEQ_LEN = 8
WV_DIM = 256
BATCH_SIZE = 1024
NTRAINSMAX = 5
EPOCHS=3
ScoreMeLi=0.8318
ScoreNetN=0.81
##########################################################################
#Python
import os
import re
import pickle
import gensim  
import time
from collections import Counter
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
from keras.layers import Concatenate, Conv1D, MaxPool1D
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten, GRU, Bidirectional
from keras.layers import Embedding, SpatialDropout1D, Activation, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
##########################################################################
NClean=0
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
    
    #Data reading & pre process
    
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
        print("Fin copia de train clean")
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
    global ScoreNetN
    ############
    NTrainsC = 0
    
    print("ScoreNetN=",ScoreNetN)
    Seed=10*(time.time()%10)
    while NTrainsC < NTRAINSMAX:
        
        print("Epochs=", EPOCHS)
        print("NetN=", NN)
        #################################        
        print("ReadTrain=",ReadTrain) #################################
        if ReadTrain:
            #Data
            print("Reading dataset train...")
            Start = np.random.randint(1, NROWS-CHUNKSIZE)
            print("Lee Train_Data desde %u a %u"%(Start, Start+CHUNKSIZE))
            Train_Data = pd.read_csv('data/trainclean.csv', skiprows = range(1, Start), nrows=CHUNKSIZE)
            Train_Data.dropna(inplace=True)
            print("Elementos de training = %u" % ( len(Train_Data) ) )
        #################################        
        #Categories
        print("DoCats=",DoCats) #################################
        if DoCats:
            print("Category Codes...")
            print("Convert as type...")
            Cats = Train_Data.category.astype('category')
            print("Get dict cats...")
            DictCats = dict( enumerate(Cats.cat.categories ) )
            print("Calc unique cats...")
            Labels = pd.unique(Train_Data.category.values)
            
            print("Save cats...")
            DictKeys={}
            for key,val in DictCats.items():
                DictKeys[val]=key

            with open("DictKeys.pkl","wb") as Handle:
                pickle.dump(DictKeys, Handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("DictCats.pkl","wb") as Handle:
                pickle.dump(DictCats, Handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("Labels.pkl","wb") as Handle:
                pickle.dump(Labels, Handle, protocol=pickle.HIGHEST_PROTOCOL)    
        else:
            with open('DictCats.pkl', 'rb') as Handle:
                DictCats = pickle.load(Handle)
            with open('DictKeys.pkl', 'rb') as Handle:
                DictKeys = pickle.load(Handle)
            with open('Labels.pkl', 'rb') as Handle:
                Labels = pickle.load(Handle)
        ##################################################################
        print("DoToken=", DoToken) #################################
        if DoToken:
            print("Tokenizer...") 
            DataTokenizer = Tokenizer(char_level=False, num_words=50000)
            print("Fitting tokenizer...") 
            DataTokenizer.fit_on_texts( Train_Data.title.values)
            DataTokenizer.num_words=50000
            with open('Tokens.pkl', 'wb') as handle:
                pickle.dump(DataTokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('Tokens.pkl', 'rb') as Handle:
                DataTokenizer = pickle.load(Handle)
        Vocab_Size = len(DataTokenizer.word_index) + 1
        print('Hay %s tokens.' % Vocab_Size)    
        ##################################################SPLIT SEQ OK
        print("GetSeqPadSplit=",GetSeqPadSplit) #SEQ SPLIT OK
        if GetSeqPadSplit: #################################
            print("Category replace...")
            Train_Data["category"].replace(DictKeys, inplace=True )
            print("To categorical...")
            Y_Enc = to_categorical(Train_Data['category'].values)
            if UseTokens:
                print("Data to seq...")
                Train_Data_Seq = DataTokenizer.texts_to_sequences(Train_Data.title.values)
                print("Seq padding...")
                Train_Data_Pad = pad_sequences(Train_Data_Seq, maxlen=MAX_SEQ_LEN)
            if UseWVec:
                Sequences = [ [Word_Index.get(t, 0) for t in Titulo] for Titulo in Train_Data.title.values ] #PRUEBA
                Train_Data_Pad = pad_sequences(Sequences, maxlen=MAX_SEQ_LEN, padding="pre", truncating="post") #PRUEBA
            
            print("Splitting dataset...")
            XTrain, XTest, YTrain, YTest = train_test_split(Train_Data_Pad, Y_Enc, test_size=0.2, random_state = 42, shuffle=True)                            
            
            # print("Guardando X/Y Train/Test Seqs")
            # with open('data/XTrain.np', 'wb') as handle:
               # pickle.dump(XTrain, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open('data/XTest.np', 'wb') as handle:
               # pickle.dump(XTest, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open('data/YTrain.np', 'wb') as handle:
               # pickle.dump(YTrain, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open('data/YTest.np', 'wb') as handle:
               # pickle.dump(YTest, handle, protocol=pickle.HIGHEST_PROTOCOL)
               
        else:
            print("Abriendo X Train Seqs...")
            with open('data/XTrain.np', 'rb') as Handle:
                XTrain = pickle.load(Handle)
            print("Abriendo X Test Seqs...")
            with open('data/XTest.np', 'rb') as Handle:
                XTest = pickle.load(Handle)
            print("Abriendo Y Train Seqs...")
            with open('data/YTrain.np', 'rb') as Handle:
                YTrain = pickle.load(Handle)
            print("Abriendo Y Test Seqs...")
            with open('data/YTest.np', 'rb') as Handle:
                    YTest = pickle.load(Handle)

        NLABELS = len ( Labels )
        print("Hay %u labels " % NLABELS)
        
        ###############################
        if UseWVec:
            NB_Words = min(MAX_NB_WORDS, len(WVModel.wv.vocab))
            print("NB_Words=",NB_Words)
            WV_Matrix = np.random.rand(NB_Words, WV_DIM)
            for word, i in Word_Index.items():
                if i >= MAX_NB_WORDS:
                    continue
                try:
                    Embedding_Vector = WVModel[word]
                    WV_Matrix[i] = Embedding_Vector
                except:
                    pass
        ###############################
        
        
        print("NN=",NN)
        #Neural Network Train / Test
        
        if NN==1:#NO TOCAR
            ModelT = Sequential() #0,0,81 OK 15 mins / 1 epoch 1.5M
            ModelT.add(Embedding(input_dim=Vocab_Size, output_dim=128, input_length=MAX_SEQ_LEN) ) 
            ModelT.add(Flatten())
            ModelT.add(Dense(256, activation='relu'))
            ModelT.add(Dense(NLABELS, activation='softmax') )
            FilePathWeights="Weights_NN1.hdf5"
        if NN==2:
            ModelT = Sequential() #0,709 15 mins/epoch/1.5M -->> 0.76 ->0.78
            ModelT.add(Embedding(input_dim=Vocab_Size, output_dim=128, input_length=MAX_SEQ_LEN) ) 
            ModelT.add(LSTM(4096, dropout=0.1))
            ModelT.add(Dense(2048, activation='relu'))
            ModelT.add(Dense(NLABELS, activation='softmax') )
            FilePathWeights="Weights_NN2.hdf5"
        if NN==3: #NO TOCAR
            ModelT = Sequential()  #0,81
            ModelT.add(Embedding(input_dim=Vocab_Size, output_dim=512, input_length=MAX_SEQ_LEN) ) 
            ModelT.add(Flatten())
            ModelT.add(Dense(2048, activation='relu'))
            ModelT.add(Dropout(0.5))
            ModelT.add(Dense(NLABELS, activation='softmax') )
            FilePathWeights="Weights_NN3.hdf5"
        if NN==4:
            ModelT = Sequential()
            ModelT.add(Embedding(input_dim=Vocab_Size, output_dim=512, input_length=MAX_SEQ_LEN))
            ModelT.add(Conv1D(128, 5, activation='relu'))
            ModelT.add(GlobalMaxPooling1D())
            ModelT.add(Dense(10, activation='relu'))
            ModelT.add(Dense(1, activation='sigmoid'))
            FilePathWeights="Weights_NN4.hdf5"
        if NN==5:  
            filters=250
            kernel_size=5
            hidden_dims=150
            ModelT = Sequential()
            ModelT.add( Embedding(input_dim=Vocab_Size, output_dim=128, input_length=MAX_SEQ_LEN) )
            ModelT.add( Dropout(0.2) )
            ModelT.add( Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1) )
            ModelT.add( GlobalMaxPooling1D() )
            ModelT.add( Dense(hidden_dims) )
            ModelT.add( Dropout(0.2), activation('relu') )
            ModelT.add( Dense(NLABELS, activation='softmax') )
            FilePathWeights="Weights_NN5.hdf5"
            
        ModelT.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])    
        ModelT.summary()
        Checkpointer = ModelCheckpoint(FilePathWeights, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        
        print("DoImportChkTrain=",DoImportChkTrain)
        if DoImportChkTrain:
            if (NTrainsC>0) or os.path.exists(FilePathWeights):
                print("Lee weigths desde:", FilePathWeights)
                ModelT.load_weights(FilePathWeights)
        
        print("DoTrainModel=", DoTrainModel)
        if DoTrainModel: ##(TRAIN)##
            print("Train model...")
            History = ModelT.fit(XTrain, YTrain, batch_size=BATCH_SIZE, verbose=1, validation_data=(XTest, YTest), shuffle=True, epochs=EPOCHS, callbacks=[Checkpointer]  )
            
        else:
            ModelT= load_model(FilePathWeights)
        
        print("DoEvaluate=", DoEvaluate)
        if DoEvaluate:
        
            print("Load model weights...")
            ModelT.load_weights(FilePathWeights)
            
            print("Evaluate model Acc...")
            Acc = ModelT.evaluate(XTest, YTest)
            print('##########Test set\n  Loss: {:0.3f}  Accuracy: {:0.3f}##########'.format(Acc[0], Acc[1]))
            Acc_A= Acc[1]
            
            print("Evaluate model B-Acc...")
            YPred = ModelT.predict(XTest, verbose=1)
            YPredMax = np.argmax(YPred, axis=1).tolist()
            YTestMax = np.argmax(YTest, axis=1).tolist()
            BAcc = balanced_accuracy_score(YTest, YPredMax)
            print ("##########BAcc: %0.3f ##########" % BAcc )
        
        print("DoTestData=", DoTestData)
        if DoTestData:
            
            if Acc_A>=ScoreNetN:
                ScoreNetN=Acc_A
                print("New score = ", ScoreNetN)
                print("Read test data")
                Test_Data = pd.read_csv('data/testclean.csv')
                Test_Data = Test_Data.fillna(' ')
                print("Elementos de test clean = %u " % ( len(Test_Data) ) )
                
                if UseWVec:
                    print("Data to sequences...")
                    TestSequences = [ [Word_Index.get(t, 0) for t in Titulo] for Titulo in Test_Data.title.values ] #PRUEBA
                    Test_Data_Pad = pad_sequences(TestSequences, maxlen=MAX_SEQ_LEN, padding="pre", truncating="post") #PRUEBA
                
                if UseTokens:
                    print("Data to sequences...")
                    Test_Data_Seq = DataTokenizer.texts_to_sequences(Test_Data.title.values)
                    print("Sequences padding...")
                    Test_Data_Pad = pad_sequences(Test_Data_Seq, maxlen=MAX_SEQ_LEN)

                print("Test predict...")
                Test_Res_Prob = ModelT.predict(Test_Data_Pad, verbose=1, batch_size=BATCH_SIZE)
                print("Test arg max...")
                Test_Res_Int = np.argmax(Test_Res_Prob, axis=1).tolist()
                print("Fin!")
               
                Test_Res =  [ DictCats[C] for C in Test_Res_Int ] 
                print("Elementos de Test_Res = %u " % ( len(Test_Res) ) )
            
                print("DoSubmit=", DoSubmit)
                if DoSubmit:
                    Test_Data.insert(3, "category", Test_Res)
                    Test_Data['category'] = Test_Data['category'].str.upper() 
                    _S = np.random.randint(low=0, high=10000)
                    print( Test_Data[ _S:_S+10 ] )
                    Submit = Test_Data[["id","category"]]
                    Submit.to_csv('data/xtian.csv', index=False)
                    print("Submit guardado")
            else:
                print("No se hace submit porque ScoreNetN(%f)<Acc(%f)"%(ScoreNetN, Acc_A))
        
        
        NTrainsC+= 1
        print("NTrainsC=",NTrainsC,"/",NTRAINSMAX)
        
    return
##########################################################################
if DoPreProc:
    PreProcess()

if DoModel:
    ModelNLP()

##########################################################################

