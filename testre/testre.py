import re
import unidecode

F = open("testre.txt", encoding="utf8")

while True:

    text = F.readline()
    
    text = text.lower() # lowercase
    StopML="de|en|em|para|con|i|sin|a|y|al|la|por|el|com|do|by|promo|envio|producao|cm|mm|oferta|producto|cuotas|interes|oportunidad"
    REPLACE_STOP = re.compile("\\b("+StopML+")\\b", re.I)
    REPLACE_SYMBOLS = re.compile('[\/(){}\[\]\|@,.\~\':;\-*\_!*+®°%²#$\"]')
    REPLACE_1LETTER = re.compile(" [a-z] {1}")
    LEAVE_ONLYCHARS = re.compile('[^a-z ]')
    REPLACE_BAD_SPACE = re.compile(' {2,}')

    text = unidecode.unidecode(text)
    #text = REPLACE_SYMBOLS.sub(' ', text)
    text = LEAVE_ONLYCHARS.sub(' ', text)
    text = REPLACE_1LETTER.sub(' ', text) 
    #text = REPLACE_ZAO.sub("cion", text)
    text = REPLACE_STOP.sub(" ", text)
    text = REPLACE_BAD_SPACE.sub(' ', text)
    print(text)
    
    input()