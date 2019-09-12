##########################################################################
# Cristian Belussi. 2019. 
# MeLi Challenge. 
##########################################################################
####
import gzip
import gensim 
import logging
import pandas as pd
import re
import unidecode

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

df = pd.read_csv('data/train.csv', nrows=20000000)
df = df["title"]

##################################
def Clean(Text):
    Text = Text.lower() # lowercase
    StopML="de|en|em|para|con|i|sin|a|y|al|la|por|el|com|do|by|promo|envio|producao|cm|mm|oferta|producto|cuotas|interes|oportunidad|promocao|gratis"
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
    
    return Text
#########################################

#for TLine in df:
#    try:
#        print(Clean(TLine))
#        input()
#    except KeyboardInterrupt:
#        break

#Clean
NL=0
AveSeq=0
Seqs=[]
for TLine in df:
    
    Line = Clean(TLine)
    
    ListWords = Line.split(" ")
    
    Seqs.append(ListWords)
    
    AveSeq += len(ListWords)
    NL+=1
    if (NL%10000)==0:
        print("NL=",NL)
        print("AveSeq=", (AveSeq/NL))
#Clean

print("WORD2VEC")
ModelW = gensim.models.Word2Vec(Seqs, min_count=5, size=256, workers=3, window=8, sg=1)
ModelW.save("model.w2v")

print("VEC TEST")
KWords=["computadora", "bicicleta", "ford","usb","toalla","celular", "mesa", "moto", "audi", "lampara","silla","taza","monitor","guitarra"]
for W_ in KWords:
    print("###################"+W_+"####################")
    try:
        for R in ModelW.wv.most_similar_cosmul(positive=[W_]):
            print(R)
    except:
        print("?")
    print("########################################")
    
    
#
