# MeLiCha2019

https://ml-challenge.mercadolibre.com/

Mi aporte para el MeLiChallenge 2019.

A medida que emprolije el código voy a comentarlo y subir cambios.

Utiliza: Python + Keras 
+ TensorFlow + SciKitLearn

Datos train/test:
Aportados por MercadoLibre en formato csv.

La idea del repositorio es que alguien que nunca haya hecho Machine Learning con textos (Hola!XD) tenga un lugar de donde empezar. Hay que usar y editar model.py para generar uno o varios modelos diferentes. Después cuando se decide que los modelos están listos, usarlos en ensemble.py que toma varios modelos y aplica todos a los mismos datos para después promediar los resultados y mejorar la predicción. En model.py hay varios flags para decidir que se quiere hacer en el script, en cada etapa del desarrollo. El código debería estar hecho con clases y separado en módulos/funcs, pero no tuve tiempo y no sabía como crecen este tipo de programas. Hay mucho hecho sobre la marcha.

En una etapa de preparación se leen los dataset tal como los entrega ML, se procesan con regex sencillas, y se almacenan en disco. No hago split de idioma por ahora para mantelerlo simple, aunque es una idea claramente interesante de probar para mejorar. No trabaje con la columna reliable del train-set porque cuando se exploran los datos solo hay un ~6% (chequar) de labels confiables.

Después de preparar los datos se pasa a la parte de lectura para training. Las palabras del titulo pasan por un tokenizer y un padding para armar el vector de entrada, o sea pasar de una oracion de texto a un vector con numeros de largo fijo. El largo promedio de titulo despues de limpiar para todo el dataset es 6, y después de algunas pruebas/error el largo 8 funciona bien. Se podría subir para mejorar alguna centesima. El proceso con el tokenizer se debería hacer solo una vez con el dataset completo, y luego siempre leerlo desde disco. Lo mismo sucede para el diccionario de categorias.

Por ahora hay un par de modelos muy elementales que pueden llevar el puntaje a 0.8 aproximadamente, de forma individual. Parece que en este tipo de problemas, textos cortos y pocos complejos, funcionan relativamente bien. Fueron entrenados con una placa de video de gama baja, aproximadamente 1 hora por cada modelo. Se pueden escalar pero se necesita más potencia de procesamiento/tiempo.
Para entrenar con todo el dataset de 20M de filas tuve problemas de memoria, así que el training esta anidado en un lazo por epocas y uno más externo para iterar sobre distintas zonas del train dataset.

#### Cosas interesantes para mejorar ####

* Reescribir el codigo con clases y módulos.

* Agregar un modulo para explorar el dataset con graficos y reportes estadísticos.

* Procesar con una mejor estrategia los titulos en español y portugues.

* Hacer algo con la columna label_quality. Los validados son una minoría.

* Generar un vector de word embedding en base al corpus del conjunto de publicaciones.

* Agregar otros modelos como NB y SVM.

* Machine Learning Wisdom?

