##########################################################################
# Cristian Belussi. 2019. 
# github.com/xtianhb/MeLiCha2019
# Licencia MIT.
# MeLi Data Challenge.
##########################################################################

import re
import unidecode

#Este simple script de python es para probar diferentes resultados
#de hacer un simple regex sobre los string de la publicacion de ML.
#Para empezar me resulto mas facil que aplicar librerias mas complejas.
# En terminos de performance y eficacia fue bastante efectiva esta estrategia.

#Archivo con varias lineas copiadas del dataset train entregado por ML.
#F = open("train.txt", encoding="utf8")
#Archivo con varias lineas copiadas del dataset test entregado por ML.
F = open("test.txt", encoding="utf8")

print(F.readline())

while True:
    try:
        Text = F.readline().split(",")[1]
        
        print("ANTES->",Text)
        
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
        
        print("DESPUES->",Text)
        
        input()
    except:
        break
        
print("Fin...")