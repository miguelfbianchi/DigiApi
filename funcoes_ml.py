
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

''' Função que executa o Grid Search dos modelos.
    Recebe como parâmetro nome do modelo, modelo base, a grade de parâmetros para avaliar e conjunto de treino
    Retorna ao final o modelo com as melhores configurações
'''
def procura_melhores_parametros (nome_modelo, modelo, parametros, X_treino, y_treino):

    # Separa uma amostra de somente 9000 entradas para teste, para o Grid Search rodar mais rápido
    _, X_treino_gs, _, y_treino_gs = train_test_split(
    X_treino, y_treino, test_size=9000, stratify=y_treino, random_state=42
)
    # Configura o Grid Search com Validação Cruzada de 3 folds
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,  # Usa todos os núcleos do processador em paralelo
        verbose=3,
    )

    # Executa a busca
    print(f"\nIniciando a busca de parâmetros otimizados do modelo {nome_modelo}...")
    inicio = time.time()
    grid_search.fit(X_treino_gs, y_treino_gs)
    fim = time.time()

    print(f"\nGrid Search concluído em {(fim - inicio)/60:.2f} minutos")

    print(f"Melhor combinação encontrada: {grid_search.best_params_}")
    print(f"Melhor Acurácia de Validação Cruzada: {grid_search.best_score_:.4f}")

    # Retorna o melhor classificador desta família
    return grid_search.best_estimator_


''' Função que treina o modelo em todo o conjunto de treino e faz as predições e métricas na base de teste
    Recebe como parâmetros o nome do classificador, o modelo já instanciado, e os conjuntos de treino e teste
    Retorna um dicionário com o nome do modelo e suas métricas, a matriz-confusão e a instância do classificacdor treinado
'''
def treina_modelo(nome_classificador, classificador, Xtreino, ytreino, Xteste, yteste):

    classificador_treinado = clone(classificador) # fundamental usar a função clone, para começar um modelo com pesos zerados.
    print(f'\nIniciando o treinamento final do modelo {nome_classificador} com toda a base de treino')
    inicio_treino = time.time()
    classificador_treinado.fit(Xtreino, ytreino)
    fim_treino = time.time()
    tempo_treino = fim_treino - inicio_treino
    print(f"Treinamento concluído com sucesso em {tempo_treino/60:.2f} minutos!")

    print('Calculando predições nos dados de treino')
    y_pred_train = classificador_treinado.predict(Xtreino)
    print('Calculando predições nos dados de teste')
    y_pred_test = classificador_treinado.predict(Xteste)

    acuracia_treino = classification_report(ytreino,y_pred_train, output_dict=True, zero_division=0)['accuracy']
    acuracia_teste = classification_report(yteste,y_pred_test, output_dict=True, zero_division=0)['accuracy']
    print (f'Acurácia treino: {acuracia_treino}')
    print (f'Acurácia teste: {acuracia_teste}')
    relatorio_metricas = classification_report(yteste,y_pred_test, output_dict=True, zero_division=0)['weighted avg']
    print (f'Métricas do modelo: \n{relatorio_metricas}')
    matriz_confusao = confusion_matrix(yteste, y_pred_test)

    return {
        'nome_modelo':nome_classificador,'acuracia':acuracia_teste,
        'precision':relatorio_metricas['precision'],'recall':relatorio_metricas['recall'],
        'f1-score': relatorio_metricas['f1-score'], 'tempo_treino':tempo_treino
        }, matriz_confusao, classificador_treinado


'''
Função que plota a matriz de confusão no formato de mapa de calor.
Recebe uma matriz numérica e o nome do modelo
'''

def plota_matriz_confusao(matriz_confusao, nome_modelo):
 
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matriz_confusao, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, linecolor="silver", cbar=True, 
        xticklabels=list(range(10)), yticklabels=list(range(10)),
    )


    plt.title(
        f"Matriz de Confusão - Modelo {nome_modelo}\nProjeto DigIA",
        fontsize=14, pad=20, fontweight="bold",
        )
    plt.xlabel("Dígito Predito pelo Modelo", fontsize=12, labelpad=10)
    plt.ylabel("Dígito Real", fontsize=12, labelpad=10)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return