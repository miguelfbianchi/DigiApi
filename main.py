# Importação das bibliotecas do ecossistema FastAPI
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Importação dos modulos da API
from routers import router_inferencia
from services import carregar_modelo

'''
Uso de decorador gerenciador de contexto assíncrono.
Configura o ambiente antes do sistema iniciar e garante seu encerramento
e limpeza de memória após finalização.
Contexto assincrono permite múltiplas requisições sem travamento da API
'''
@asynccontextmanager
async def ciclo_de_vida (app: FastAPI):
    # Roda este trecho somente uma vez no carregamento do sistema (até o yield)
    print ("Iniciando o servidor...carregando modelo")

    # Carregamento na memória do modelo de ML
    modelo_treinado = carregar_modelo()
    # Grava o modelo no estado global do sistema
    app.state.modelo = modelo_treinado

    # Coloca o sistema oficialmente "em pé"
    # Pausa a execução do bloco até o sistema encerrar
    yield

    # Limpa a memória quando o servidor Uvicorn for encerrado
    print ("Desligando o servidor e limpando a memória")
    del app.state.modelo

# Instanciamento da API já com as características do ciclo de vida definidas (lifespan)
app = FastAPI(
    title = "API de Reconhecimento de dígitos",
    description= "Serviço preditivo para reconhecimento de imagens de dígitos manuscritos",
    lifespan = ciclo_de_vida
)


# Acopla o roteador modularizado na isntância principal da aplicação
app.include_router(router_inferencia)

# Define uma rota base de apresentação utilizando GET na raiz da API
@app.get("/")
def home():
    return {
        "status": "Online",
        "documentacao": "Acesse /docs no navegador para testar a interface do Swagger."
    }