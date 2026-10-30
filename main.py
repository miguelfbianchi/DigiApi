from fastapi import FastAPI
from routers import router_inferencia

app = FastAPI(
    title = "API de Reconhecimento de dígitos",
    description= "Serviço preditivo para reconhecimento de imagens de dígitos manuscritos"
)

app.include_router(router_inferencia)

@app.get("/")
def home():
    return {
        "status": "Online",
        "documentacao": "Acesse /docs no navegador para testar a interface do Swagger."
    }