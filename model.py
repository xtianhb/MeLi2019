##########################################################################
# Cristian Belussi. 2019. 
# github.com/xtianhb/MeLiCha2019
# Licencia MIT.
# MeLi Data Challenge.
##########################################################################
DoPreProc=0  #Habilitar leer, limpiar, y guardar datos
PreReadTrain=0 #Habilitar para train
PreReadTest=0 #Habilitar para test
CleanTrain=0 #Limpiar train
CleanTest=0 #Limpiar test
#################################
DoModel=1 #Procesar modelo
ReadTrain=1 #Leer csv procesado
DoCats=0 #Calcular categorias. Se deberia hacer 1 vez en todo el dataset de train.
DoToken=0 #Calcular tokens. Se deberia hacer 1 vez en todo el dataset de train.
GetSeqPadSplit=1 #Hace un split de datos para train/test para los modelos.
DoTrainModel=1 #Entrenar modelos
DoImportChkTrain=1     #Importar si hay weights en disco para el modelo
DoEvaluate=1 #Procesar test data, predict challenge
DoSubmit=1  #Generar archivo submit
##########################################################################
NN=2 #Modelo a entrenar
NROWS=20000000 #Cantidad de filas a leer cuando se procesa el train dataset
CHUNKSIZE=1875000 #Cantidad de filar para entrenar. (Para problemas de memoria)
MAX_SEQ_LEN = 8 #Cantidad de palabras maximas del title ya pre procesado.
BATCH_SIZE = 1024 #Dimension del batch size, para usar con GPU
NTRAINSMAX = 5 # Num de ciclos que repetimos por Epocas con CHUNKSIZE datos.
EPOCHS=3 #Numero de epocas. Aqui es veces que entrenamos con CHUNKSIZE datos.
ScoreMeLi=0.8318 #Para decidir guardar, puntaje actual global
ScoreNetN=[0.81, 0.81, 0.81, 0.81, 0.81] #puntaje actual de las redes
NClean=0#Contador
##########################################################################
#Python libraries
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
from keras.layers.merge import add
from keras.utils import np_utils, to_categorical
from keras.layers import Concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten, Bidirectional
from keras.layers import Embedding, Activation, TimeDistributed, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
##########################################################################
def Clean_Text(Text):
    global NClean   
    
    Text = Text.lower() # lowercase
    #Palabras que no aportan al contenido del titulo
    StopML="de|en|em|para|con|i|sin|a|y|al|la|por|el|com|do|by|promo|envio|producao|cm|mm|oferta|producto|cuotas|interes|oportunidad"
    #Reglax regex
    #Borra palabras stop
    REPLACE_STOP = re.compile("\\b("+StopML+")\\b")    
    #Solo queda letras del abecedario
    LEAVE_ONLYCHARS = re.compile('[^a-z ]')
    #Borra letras sueltas
    REPLACE_1LETTER = re.compile("\\b[a-z ]\\b")
    #Borra palabras de 2 letras
    REPLACE_2LETTER = re.compile("\\b[a-z]{2}\\b")
    #Borra espacios
    REPLACE_BAD_SPACE = re.compile('\s+')

    #Convierte todo a ascii
    Text = unidecode.unidecode(Text)
    #Borra numeros y signos
    Text = LEAVE_ONLYCHARS.sub(' ', Text)
    #Borra letras sueltas
    Text = REPLACE_1LETTER.sub(' ', Text) 
    #Borra 2 letras
    Text = REPLACE_2LETTER.sub(' ', Text) 
    #Borra palabras irrelevantes
    Text = REPLACE_STOP.sub(" ", Text)
    #Borra espacios
    Text = REPLACE_BAD_SPACE.sub(' ', Text)
    
    #Debug para ir viendo como va el cleaning
    NClean+=1
    if(NClean%10000)==0:
        print("N=%u  values"%NClean, end="\r")
    
    return Text
##########################################################################
##########################################################################
def PreProcess():
    
    #Data reading & pre process
    
    global NClean #Debug
    
    if PreReadTrain: #Tal vez solo queremos procesar training set
        # DataSet Train columnas: 'title', 'label_quality', 'language', 'category'
        print("Leyendo dataset train...") #Lee dataset training
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
            #Aplica la funcion para limpiar el titulo de la publicacion
            Train_Data['title'] = Train_Data['title'].apply(Clean_Text)
        print("Elementos de training limpios = %u " % ( len(Train_Data) ) )
        print("TRAIN DESPUES DE LIMPIAR:")
        print(RawTrain.columns)
        print(Train_Data[0:10])
        print("Guarda copia de train clean")
        Train_Data.to_csv('data/trainclean.csv', index=False, na_rep=" ")
        print("Fin copia de train clean")
        
    if PreReadTest==1: #Tal vez solo queremos procesar testing set
        # Dataset Test columnas: 'id', 'title', 'language'
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
def ModelNLP():

    global ScoreNetN
    
    ############
    NTrainsC = 0 #Contador de trainings con diferentes chunks de train data
    
    print("ScoreNetN=",ScoreNetN[NN])
    Seed=10*(time.time()%10)
    
    # Vamos a entrenar el modelo NTRAINSMAX veces con diferentes CHUNKSIZE datos
    # Cada loop hace EPOCHS epocas.
    # Es una forma simple de entrenar con todo el train dataset y no tener problemas de memoria.
    while NTrainsC < NTRAINSMAX:
        
        print("Epochs=", EPOCHS)
        print("NetN=", NN)
        #################################        
        print("ReadTrain=",ReadTrain) #################################
        if ReadTrain:
            #Data
            print("Reading dataset train...")
            #Selecciona un lugar pseudo aleatorio del dataset
            Start = np.random.randint(1, NROWS-CHUNKSIZE)
            print("Lee Train_Data desde %u a %u"%(Start, Start+CHUNKSIZE))
            Train_Data = pd.read_csv('data/trainclean.csv', skiprows = range(1, Start), nrows=CHUNKSIZE)
            #Eliminar filas invalidas
            Train_Data.dropna(inplace=True)
            print("Elementos de training = %u" % ( len(Train_Data) ) )
        #################################        
        #Categories
        print("DoCats=",DoCats) #################################
        #Esto deberia hacerse solo una vez en todo el Train Dataset.
        #Busca todas las categorias(labels) posibles de producto y las vectoriza.
        #Despues en training del modelo esto esta desactivado.
        if DoCats:
            print("Category Codes...")
            print("Convert as type...")
            # category.astype un tipo especial que maneja pandas
            Cats = Train_Data.category.astype('category')
            print("Get dict cats...")
            #Armamos un diccionario de python, nos va servir despues.
            #Este diccionario se usa DictCats[Categoria] -> ClaveNum
            #Pasamos de un texto a un numero.
            DictCats = dict( enumerate(Cats.cat.categories) )
            print("Calc unique cats...")
            #Tenemos una lista de Labels
            Labels = pd.unique(Train_Data.category.values)
            
            print("Save cats...")
            #Armamos un diccionario al reves que el anterior.
            #Este diccionario se usa DictCats[ClaveNum] -> Categoria.
            #Pasamos de un numero a un texto.
            DictKeys={}
            for key,val in DictCats.items():
                DictKeys[val]=key
            
            #Guarda los diccionarios y categorias.
            with open("DictKeys.pkl","wb") as Handle:
                pickle.dump(DictKeys, Handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("DictCats.pkl","wb") as Handle:
                pickle.dump(DictCats, Handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open("Labels.pkl","wb") as Handle:
                pickle.dump(Labels, Handle, protocol=pickle.HIGHEST_PROTOCOL)    
        else:
            #Despues simplemente lo leemos desde el disco.
            with open('DictCats.pkl', 'rb') as Handle:
                DictCats = pickle.load(Handle)
            with open('DictKeys.pkl', 'rb') as Handle:
                DictKeys = pickle.load(Handle)
            with open('Labels.pkl', 'rb') as Handle:
                Labels = pickle.load(Handle)
        ##################################################################
        print("DoToken=", DoToken) #################################
        #Al modelo hay que alimentarlo con numeros. 
        #Hay que convertir todas las posibles palabras a un token unico.
        #Esta funcion es para eso, cada palabra unica tiene un id unico.
        #Es una representacion/mapeo de string a integer y afecta la prediccion del modelo.
        if DoToken: #Los tokens los calculamos al principio una vez con el Train Dataset
            print("Tokenizer...") 
            DataTokenizer = Tokenizer(char_level=False, num_words=50000)
            print("Fitting tokenizer...") 
            DataTokenizer.fit_on_texts( Train_Data.title.values)
            DataTokenizer.num_words=50000
            with open('Tokens.pkl', 'wb') as handle:
                pickle.dump(DataTokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            #Lo calculamos una vez, y luego siempre:
            with open('Tokens.pkl', 'rb') as Handle:
                DataTokenizer = pickle.load(Handle)
        #Cantidad en nuestro universo de palabras
        Vocab_Size = len(DataTokenizer.word_index) + 1
        print('Hay %s tokens.' % Vocab_Size)    
        ##################################################SPLIT SEQ OK
        print("GetSeqPadSplit=",GetSeqPadSplit) #SEQ SPLIT OK
        if GetSeqPadSplit: #################################
            #Esta funcion terminar de procesar los datos para llevarlos a un formato entendible.
            #El modelo es alimentado con un vector con MAX_SEQ_LEN tokens(palabras)
            #Si es mas corto o mas largo, se rellena o recorta.
            print("Category replace...")
            Train_Data["category"].replace(DictKeys, inplace=True )
            print("To categorical...")
            Y_Enc = to_categorical(Train_Data['category'].values)
            print("Data to seq...")
            Train_Data_Seq = DataTokenizer.texts_to_sequences(Train_Data.title.values)
            print("Seq padding...")
            Train_Data_Pad = pad_sequences(Train_Data_Seq, maxlen=MAX_SEQ_LEN)
            
            #Esta funcion toma el train data set procesado, y lo separa en train/test
            print("Splitting dataset...")
            XTrain, XTest, YTrain, YTest = train_test_split(Train_Data_Pad, Y_Enc, test_size=0.20, random_state = 42, shuffle=True)                            
            
            #Esto es util por si queremos hacer pruebas con un train/test set constante guardado en disco.
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
            #Cuando queremos leer un train/test set guardado en disco.
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
        
        print("NN=",NN)
        #Neural Network Train / Test
        
        #Una vez que tenemos un modelo con buena performance, no lo modificamos.
        #Lo vamos a copiar y usar en el modulo ensamble.
        #ToDo: Mejorar este pipeline con clases y etc...
        
        #NN1
        #Hasta ahora descubri que modelos simples con NN funcionan bien, i.e. ValAcc=0.8
        if NN==1:
            ModelT = Sequential() #Val_Acc ~ 0.81
            ModelT.add(Embedding(input_dim=Vocab_Size, output_dim=128, input_length=MAX_SEQ_LEN) ) 
            ModelT.add(Flatten())
            ModelT.add(Dense(256, activation='relu'))
            ModelT.add(Dense(NLABELS, activation='softmax') )
            FilePathWeights="Weights_NN1.hdf5"
        #Los modelos con modelos mas complejos creo que necesitan muchos datos para hacer la diferencia.
        if NN==2:
            ModelT = Sequential() #Val_Acc ~ ?
            ModelT.add(Embedding(input_dim=Vocab_Size, output_dim=128, input_length=MAX_SEQ_LEN) ) 
            ModelT.add(Conv1D(256, 4, activation='relu'))
            ModelT.add(GlobalMaxPooling1D())
            ModelT.add(Dense(NLABELS, activation='softmax'))
            ModelT.summary()
            FilePathWeights="Weights_NN2.hdf5"
        if NN==3: 
            ModelT = Sequential()  #Val_Acc ~ 0.81
            ModelT.add(Embedding(input_dim=Vocab_Size, output_dim=512, input_length=MAX_SEQ_LEN) ) 
            ModelT.add(Flatten())
            ModelT.add(Dense(2048, activation='relu'))
            ModelT.add(Dropout(0.5))
            ModelT.add(Dense(NLABELS, activation='softmax') )
            FilePathWeights="Weights_NN3.hdf5"
        if NN==4:
            ModelT = Sequential() #Val_Acc ~ ?
            ModelT.add(Embedding(input_dim=Vocab_Size, output_dim=256, input_length=MAX_SEQ_LEN))
            ModelT.add(Bidirectional(LSTM(256, return_sequences=True)))
            ModelT.add(TimeDistributed(Dense(NLABELS,activation='softmax') ) )
            FilePathWeights="Weights_NN4.hdf5"
       
        #Compila el modelo seleccionado y muestra debug
        ModelT.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])    
        ModelT.summary()
        #Esta funcion (hay otras) muestra un debug durante el entrenamiento. Viene con Keras.
        Checkpointer = ModelCheckpoint(FilePathWeights, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        
        #Cuando queremos mejorar un modelo con mas datos, leemos los weights desde el disco,
        #y lo seguimos mejorando con datos nuevos.
        print("DoImportChkTrain=",DoImportChkTrain)
        if DoImportChkTrain:
            try:
                if (NTrainsC>0) or os.path.exists(FilePathWeights):
                    print("Lee weigths desde:", FilePathWeights)
                    ModelT.load_weights(FilePathWeights)
                except:
                    print("Error cargando weights", FilePathWeights)
        
        #Aqui nos detenemos a entrenar el modelo, o solo leerlo.
        print("DoTrainModel=", DoTrainModel)
        if DoTrainModel: ##(TRAIN)##
            print("Train model...")
            History = ModelT.fit(XTrain, YTrain, batch_size=BATCH_SIZE, verbose=1, validation_data=(XTest, YTest), shuffle=True, epochs=EPOCHS, callbacks=[Checkpointer]  )
        
        #Esta parte hace una evaluacion del modelo con test set que armamos antes.
        print("DoEvaluate=", DoEvaluate)
        if DoEvaluate:
        
            print("Load model weights...")
            #Carga los weights del modelo. Borrar de aqui?
            ModelT.load_weights(FilePathWeights)
            
            print("Evaluate model Acc...")
            #Calculamos el Validation Accuracy.
            #Como nos indican en el challenge, si el dataset es desbalanceado en cantidad de clases,
            #esta metrica no es tan representativa.
            Acc = ModelT.evaluate(XTest, YTest)
            print('##########Test set\n  Loss: {:0.3f}  Accuracy: {:0.3f}##########'.format(Acc[0], Acc[1]))
            Acc_A= Acc[1] #Guarda ValAcc
            
            #Segun las reglas, van a usar el B_Acc
            #El Balanced Accuracy, es un promedio del Recall a traves de todas las clases.
            #O sea, calcula el procentaje que acertamos cada clase, y lo promedia el de las N Clases.
            print("Evaluate model B-Acc...")
            YPred = ModelT.predict(XTest, verbose=1)
            #argmax reduce el vector de resultados (probabilidades de cada Cat.) al valor maximo
            YPredMax = np.argmax(YPred, axis=1).tolist()
            YTestMax = np.argmax(YTest, axis=1).tolist()
            BAcc = balanced_accuracy_score(YTestMax, YPredMax)
            print ("##########BAcc: %0.3f ##########" % BAcc )
        
            #Check Negatives
            #Este codigo esta en desarrollo. Es para explorar los casos con error.
            NNeg=0
            Idx=0
            while NNeg<100:
                try:
                    CaN = Train_Data.category.values[Idx]
                    CaNP = YPredMax[Idx]
                    if int(CaN) != int(CaNP):
                        print(Idx, Train_Data.title.values[Idx], DictCats[CaN], DictCats[CaNP])
                        NNeg+=1
                    Idx+=1
                except:
                    print("Err negatives")
                    break
                
            print("Fin analisis...")
        
        #Esta parte es para procesar los datos para el submit, el test set original.
        print("DoTestData=", DoTestData)
        if DoTestData:
            #Decidir si vale la pena calcularlo por el score obtenido...
            if Acc_A>=ScoreNetN[NN]:
                ScoreNetN[NN]=Acc_A #Actualizamos el score.
                print("New score = ", ScoreNetN[NN])
                
                #Leemos el test set de ML ya procesado.
                print("Read test data")
                Test_Data = pd.read_csv('data/testclean.csv')
                Test_Data = Test_Data.fillna(' ')
                print("Elementos de test clean = %u " % ( len(Test_Data) ) )
                
                #Hacemos lo mismo que antes, convertir un titulo con palabras
                #en un vector con numero, de largo fijo.
                print("Data to sequences...")
                Test_Data_Seq = DataTokenizer.texts_to_sequences(Test_Data.title.values)
                print("Sequences padding...")
                Test_Data_Pad = pad_sequences(Test_Data_Seq, maxlen=MAX_SEQ_LEN)

                #Hace las predicciones con el modelo que armamos, con los datos de test.
                print("Test predict...")
                Test_Res_Prob = ModelT.predict(Test_Data_Pad, verbose=1, batch_size=BATCH_SIZE)
                print("Test arg max...")
                ##argmax reduce el vector de resultados (probabilidades de cada Cat.) al valor maximo
                Test_Res_Int = np.argmax(Test_Res_Prob, axis=1).tolist()
                print("Fin!")
               
                #Usamos el diccionario para recuperar los nombres de categoria.
                Test_Res =  [ DictCats[C] for C in Test_Res_Int ] 
                print("Elementos de Test_Res = %u " % ( len(Test_Res) ) )
            
                #Armamos el CSV para submit a la pagina.
                print("DoSubmit=", DoSubmit)
                if DoSubmit:
                    Test_Data.insert(3, "category", Test_Res)
                    Test_Data['category'] = Test_Data['category'].str.upper() 
                    #Imprime un rango de valores partiendo de una fila aleatoria
                    _S = np.random.randint(low=0, high=50000)
                    print( Test_Data[ _S:_S+25 ] )
                    Submit = Test_Data[["id","category"]]
                    Submit.to_csv('data/xtian.csv', index=False)
                    print("Submit guardado!")
                    print("Buena suerte!")
            else:
                print("No se calcula submit porque ScoreNetN(%f)<Acc(%f)"%(ScoreNetN[NN], Acc_A))
        
        #Loop principal de trianing de un modelo. EPOCHS & CHUNKSIZE
        NTrainsC+= 1
        print("NTrainsC=",NTrainsC,"/",NTRAINSMAX)
        
    return
##########################################################################
if DoPreProc:
    PreProcess()

if DoModel:
    ModelNLP()

##########################################################################

