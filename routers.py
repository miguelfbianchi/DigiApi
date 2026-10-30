from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from services import carregar_imagem_upload, processa_digito_manual, executar_inferencia

router_inferencia = APIRouter(
    prefix = '/services',
    tags = ['Inspeção e Inferência']
)

@router_inferencia.post ('/inferencia')
async def reconhecer_digito(request: Request, file: UploadFile = File(...)):
    
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail = "Apenas imagens PNG ou JPEG !")

    conteudo = await file.read()

    try:
        imagem = carregar_imagem_upload (conteudo)
        imagem_inferir = processa_digito_manual(imagem)

        modelo = request.app.state.modelo
        resultado_final = executar_inferencia(modelo, imagem_inferir)
        
        return resultado_final

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento: {str(e)}")