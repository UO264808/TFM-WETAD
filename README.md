# TFM - WETAD 
## _Word Embeddings for Anomaly Detection_

Este repositorio contiene una recopilación de los experimentos desarrollados en el Trabajo Fin de Master _"Diagnóstico inteligente de anomalías y pronóstico de vida útil"_. El código se encuentra dividido en dos secciones bien diferenciadas: Una parte (markov_chain_experiments) se encuentra destinada a los experimentos realizados con series simbólicas basadas en Cadenas de Markov. La otra parte se corresponde con la comparativa realizada entre WETAD y los métodos obtenidos como alternativas del Estado del Arte.

## Dependencias
En caso de necesitar volver a repetir los experimentos será necesario instalar las correspondientes depedencias haciendo uso de entornos virtuales. En la siguiente lista aparecen las dependencias para ambas secciones;

```
pandas
tqdm
matplotlib
dgl
numpy
tensorflow
SciencePlots
scikit-learn
scipy
xlrd==1.2.0
```

## Ejecución 
Para la ejecución de los experimentos tan sólo hay que lanzar el intérprete de Python pasando como parámetro el archivo fuente a ejecutar. Por ejemplo:

```
python3 univariate_test_pairs.py
```

En el caso de los experimentos de la comparativa se realiza de manera similar. Lanzando el archivo fuente main e indicando mediante parámetros el modelo y el conjunto de datos a utilizar.
```
python3 main.py --model <model> --dataset <dataset> --retrain
```

## Autoría y licencias
El diseño y desarrollo del método WETAD y de los experimentos es propio excepto los de la comparativa, donde el código es obra de los autores de TranAD así como la implementación de los métodos utilizados en la comparativa.
El código de TranAD cuenta con la licencia BSD-3-Clause. El archivo de Licencia viene incluído.
