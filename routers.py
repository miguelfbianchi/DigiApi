from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status
from services import carregar_imagem_upload, processa_digito_manual, executar_inferencia

from schemas import InferenciaSchema, HealthSchema

# Implementa rota principal da aplicação
router_inferencia = APIRouter(
    prefix = '/services',
    tags = ['Inspeção e Inferência']
)

# Rota com o método GET que verifica se a apliacação está "em pé"
@router_inferencia.get ('/health', response_model=HealthSchema,
                        status_code=status.HTTP_200_OK
                        )
def health_check (request: Request):
    """
    Checa a integridade operacional da API e se o modelo reside na memória RAM.
    """
    # Verifica se o estado global contém um item com atributo "modelo"
    modelo_na_ram = hasattr(request.app.state, "modelo") and request.app.state.modelo is not None

    if not modelo_na_ram:
        # Se o servidor estiver de pé mas o modelo falhou ao carregar, retorna 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="O serviço está ativo, mas o modelo de Machine Learning não foi encontrado na memória RAM."
        )

    # Retorna mensagem de sucesso caso o modelo esteja na memória.    
    return {
        "status": "healthy",
        "modelo_carregado": True
    }

# Rota de inferência (principal) com método POST
# O parâmetro response_mode define o padrão de dados da resposta
# Este é definido no módulo schemas.py através do objeto InferenciaSchema
# E validado pelo Pydantic
# O dado de entrada é um arquivo de imagem, então não será validado com Pydantic
# Já que é recebido via formulário multiform. As validações são feitos no código da função
@router_inferencia.post ('/inferencia', response_model = InferenciaSchema,
                         status_code=status.HTTP_200_OK
                         )
async def reconhecer_digito(request: Request, file: UploadFile = File(...)):
    """
    Endpoint assíncrono responsável por receber um arquivo de imagem binária,
    invocar as camadas de serviço (OpenCV + Scikit-Learn) e retornar a classificação.
    
    Parâmetros:
    - request (Request): Objeto da requisição atual, usado como chave para acessar
                         o modelo de ML residente na memória RAM (app.state.modelo).
    - file (UploadFile): O arquivo binário da imagem, enviado via formulário 
                         multipart/form-data. O caractere '...' indica
                         que este campo é obrigatório.
    """
    # Verifica se o tipo de arquivo é compatível
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Tipo de arquivo inválido. O corpo da requisição deve conter uma imagem PNG ou JPEG."
        )

    try:
        conteudo = await file.read()

        # Veriica se o conteúdo do arquivo é válido
        if not conteudo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="O arquivo enviado está vazio ou corrompido."
                )

        # Transforma o conteúdo em bytes puros do upload em matriz Numpy
        imagem = carregar_imagem_upload (conteudo)
        # Pré-processamento da imagem
        imagem_inferir = processa_digito_manual(imagem)

        if imagem_inferir is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="O pipeline OpenCV não conseguiu extrair características válidas desta imagem."
            )

        modelo = request.app.state.modelo
        # Inferência propriamente dita
        resultado_final = executar_inferencia(modelo, imagem_inferir)
        
        return resultado_final

    except HTTPException as http_err:
        # Repassa os erros 400 ou 422 capturados acima sem transformá-los em 500
        raise http_err

    # Tratamento final de erros para evitar quebra da aplicação
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erro no processamento interno: {str(e)}"
        )