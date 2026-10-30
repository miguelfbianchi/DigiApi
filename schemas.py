from pydantic import BaseModel, Field

class InferenciaSchema(BaseModel):
    """
    Contrato estrito de saída para o resultado da classificação de dígitos.
    Garante que os dados matemáticos do modelo sejam validados antes do envio.
    """
    
    digito_inferido: int = Field(
        ...,
        ge=0,
        le=9,
        description="O dígito numérico manuscrito reconhecido pelo modelo (deve ser entre 0 e 9).",
        examples=[5]
    )
    
    probabilidade: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="A certeza do modelo em relação à predição, expressa em formato de porcentagem (0.0% a 100.0%).",
        examples=[98.42]
    )

    # Configuração de metadados para documentação do Swagger UI
    model_config = {
        "json_schema_extra": {
            "example": {
                "digito_inferido": 3,
                "probabilidade": 100.0
            }
        }
    }
