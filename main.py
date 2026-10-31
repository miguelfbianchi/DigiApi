from fastapi import FastAPI
from contextlib import asynccontextmanager
from routers import router_inferencia
from services import carregar_modelo

@asynccontextmanager
async def ciclo_de_vida (app: FastAPI):
    print ("Iniciando o servidor...carregando modelo")

    modelo_treinado = carregar_modelo()
    app.state.modelo = modelo_treinado

    yield

    print ("Desligando o servidor e limpando a memória")
    del app.state.modelo
    
app = FastAPI(
    title = "API de Reconhecimento de dígitos",
    description= "Serviço preditivo para reconhecimento de imagens de dígitos manuscritos",
    lifespan = ciclo_de_vida
)

app.include_router(router_inferencia)

@app.get("/")
def home():
    return {
        "status": "Online",
        "documentacao": "Acesse /docs no navegador para testar a interface do Swagger."
    }