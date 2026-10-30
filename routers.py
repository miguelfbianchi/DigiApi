from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status
from services import carregar_imagem_upload, processa_digito_manual, executar_inferencia

from schemas import InferenciaSchema, HealthSchema

router_inferencia = APIRouter(
    prefix = '/services',
    tags = ['Inspeção e Inferência']
)

@router_inferencia.get ('/health', response_model=HealthSchema,
                        status_code=status.HTTP_200_OK
                        )
def health_check (request: Request):
    """
    Checa a integridade operacional da API e se o modelo reside na memória RAM.
    """
    modelo_na_ram = hasattr(request.app.state, "modelo") and request.app.state.modelo is not None

    if not modelo_na_ram:
        # Se o servidor estiver de pé mas o modelo falhou ao carregar, retorna 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="O serviço está ativo, mas o modelo de Machine Learning não foi encontrado na memória RAM."
        )
        
    return {
        "status": "healthy",
        "modelo_carregado": True
    }


@router_inferencia.post ('/inferencia', response_model = InferenciaSchema,
                         status_code=status.HTTP_200_OK
                         )
async def reconhecer_digito(request: Request, file: UploadFile = File(...)):
    
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Tipo de arquivo inválido. O corpo da requisição deve conter uma imagem PNG ou JPEG."
        )

    try:
        conteudo = await file.read()
        
        if not conteudo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="O arquivo enviado está vazio ou corrompido."
                )

        imagem = carregar_imagem_upload (conteudo)
        imagem_inferir = processa_digito_manual(imagem)

        if imagem_inferir is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="O pipeline OpenCV não conseguiu extrair características válidas desta imagem."
            )

        modelo = request.app.state.modelo
        resultado_final = executar_inferencia(modelo, imagem_inferir)
        
        return resultado_final

    except HTTPException as http_err:
        # Repassa os erros 400 ou 422 capturados acima sem transformá-los em 500
        raise http_err

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erro no processamento interno: {str(e)}"
        )