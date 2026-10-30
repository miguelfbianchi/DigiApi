from fastapi import FastAPI

from services import services_route

app = FastAPI(
    title = "API de Reconhecimento de dígitos",
    description= "Serviço preditivo para reconhecimento de imagens de dígitos manuscritos"
)

app.include_router(services_route)