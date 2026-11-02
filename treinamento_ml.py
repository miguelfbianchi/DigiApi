import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import sqlite3
import os
from funcoes_ml import *

# Carregando dataset MNIST a partir da OpenML
print("Baixando o dataset MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
print("Finalizou o processo de baixar o dataset MNIST!")

# Separação entre features e target
X, y = mnist.data, mnist.target

# Exploração da dimensionalidade das matrizes
print (f'========>> A matriz X possui {X.shape[0]} amostras, sendo {X.shape[1]} pixels por imagem')
print (f'========>> A matriz y é unidimensional e possui {y.shape[0]} rótulos / targets\n')
print (f'========>> Matriz de pixels de uma amostra:\n\n {X[10000]}')


print(X.shape)

# Verificação da distribuição das classes e balanceamento do dataset
rotulos, freq_abs = np.unique(y, return_counts=True)
freq_perc = np.round(freq_abs / freq_abs.sum() * 100, decimals = 2)
print (f'Rótulos únicos: {rotulos}')
print (f'Frequências absolutas: {freq_abs}')
print (f'Frequências percentuais: {freq_perc}')

# Averiguação dos tipos de dados
print (f'Tipos de dados da matriz X: {X.dtype}')
print (f'Tipos de dados da matriz y: {y.dtype}')

# Alteração do tipo de dados da matriz de rótulos para inteiro
y = y.astype(np.uint8)

# divisão estratificada dos dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=42)

# escalonamento dos dados
scaler = MinMaxScaler()

X_train_escalonado = scaler.fit_transform(X_train)
X_test_escalonado = scaler.transform(X_test)

''' Cria uma lista de dicionários, no qual cada item é uma família de modelos
    Cada um recebe o nome/tipo do modelo, um dicionário com parâmetros de gridsearch e o modelo base a ser instanciado
'''
modelos_testar = []

modelos_testar.append({
    'nome_modelo':'RandomForest',
    'param_grid': {'n_estimators':[50,150], # número de árvores
                    'max_depth':[10,20]}, # profundidade da árvore
    'modelo_base': RandomForestClassifier(n_jobs=-1, random_state=42)
    })

modelos_testar.append({
    'nome_modelo':'MLP',
    'param_grid': {
        "hidden_layer_sizes": [(64,), (100, 50)], # testa uma camada de 64 layers ocultos e duas camadas de 100 e 50 layers ocultos
        "learning_rate_init": [0.001, 0.01]}, # taxa de aprendizagem inicial. Ela vai sendo alterada dinamicamente durante o treino
    'modelo_base': MLPClassifier(
        activation="relu", # função de ativação dos neurônios. Matematicamente simples e atenua a oscilação
        solver="adam", # otimizador que atualiza os pesos e vieses dinamicamente e individualmente para cada neurônio
       early_stopping=True, # para o treinamento quando ele parar de melhorar. Economiza processamento e tempo
        random_state=42,
        verbose=True)
    })


'''
    A lista de modelos vai ser percorrida e submetida à avaliação de parâmetros
    Os melhores de cada família farão parte de outra lista de modelos que serão treinados em toda a base
'''
melhores_modelos = []
for i in modelos_testar:
    nome_modelo = i['nome_modelo']
    param_grid = i['param_grid']
    modelo_base = i['modelo_base']

    melhor_modelo = procura_melhores_parametros(nome_modelo, modelo_base, param_grid, X_train_escalonado, y_train)
    melhores_modelos.append({'nome_modelo':nome_modelo, 'modelo': melhor_modelo})

'''
Treinamento dos 3 melhores modelos de cada família em toda a base de treino
Serão geradas 3 listas contendo: as métricas de cada modelo, as matrizes de confusão,
e as instâncias dos modelos treinados
'''
resultados_modelos = []
matrizes = []
modelos_treinados = []

for m in melhores_modelos:

    nome_classificador = m['nome_modelo']
    classificador = m['modelo']
    
    resultados_modelo, matriz_confusao, classificador_treinado = treina_modelo(
        nome_classificador, classificador, X_train_escalonado, y_train, X_test_escalonado, y_test)
    
    resultados_modelos.append(resultados_modelo)
    matrizes.append({'classificador':nome_classificador, 'matriz_confusao': matriz_confusao})
    modelos_treinados.append({'modelo': nome_classificador, 'modelo_treinado': classificador_treinado})


# Exibe tabela com os resultados / métricas dos 3 modelos.
metricas_modelos = pd.DataFrame(resultados_modelos)
print ("\n")
print (metricas_modelos)

# Grava arquivo json com relatório de métricas dos modelos
print ("\nGravando arquivo .json com métricas dos modelos")
os.makedirs("outputs", exist_ok=True)
metricas_modelos.to_json('outputs/resultados.json', orient = 'index')

# serialização de um dos modelos (MLP) para gravação no banco de dados local
nome_modelo_serializar = modelos_treinados[1]['modelo']
modelo_serializado = pickle.dumps(modelos_treinados[1]['modelo_treinado'])

# Conecta ao banco de dados local e cria automaticamente caso não exista
conn = sqlite3.connect("digia_banco.db")
cursor = conn.cursor()

# Consulta para criação da tabela modelos que receberá os modelos e os pesos
query = """
        CREATE TABLE IF NOT EXISTS modelos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_modelo TEXT NOT NULL,
            pesos_binarios BLOB NOT NULL
        )"""

cursor.execute(query)

# Consulta para inserir o modelo serializado no banco de dados
query = "INSERT INTO modelos (nome_modelo, pesos_binarios) VALUES (?, ?)"

cursor.execute(query, (nome_modelo_serializar, modelo_serializado))

conn.commit()
conn.close()

print ("Modelo treinado e armazenado no banco de dados")