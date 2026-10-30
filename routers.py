from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status
from services import carregar_imagem_upload, processa_digito_manual, executar_inferencia

from schemas import InferenciaSchema

router_inferencia = APIRouter(
    prefix = '/services',
    tags = ['Inspeção e Inferência']
)

@router_inferencia.post ('/inferencia', response_model = InferenciaSchema,
                         status_code=status.HTTP_200_OK
                         )
async def reconhecer_digito(request: Request, file: UploadFile = File(...)):
    
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Tipo de arquivo inválido. O corpo da requisição deve conter uma imagem PNG ou JPEG."
        )

    conteudo = await file.read()

    try:
        imagem = carregar_imagem_upload (conteudo)
        imagem_inferir = processa_digito_manual(imagem)

        modelo = request.app.state.modelo
        resultado_final = executar_inferencia(modelo, imagem_inferir)
        
        return resultado_final

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erro no processamento: {str(e)}"
        )