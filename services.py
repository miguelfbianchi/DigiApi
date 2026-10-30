from fastapi import APIRouter, UploadFile, File, HTTPException
from tratamento_imagens import *

import sqlite3
import pickle

def carregar_modelo():
    conn = sqlite3.connect("digia_banco.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT pesos_binarios FROM modelos WHERE nome_modelo = ? ORDER BY id DESC LIMIT 1", ("MLP",))
    linha = cursor.fetchone()
    conn.close()
    
    if linha is None:
        raise RuntimeError("Nenhum modelo foi encontrado na base !")
    
    pesos_binarios = linha[0]
    modelo_carregado = pickle.loads(pesos_binarios)
    return modelo_carregado

modelo = carregar_modelo()

services_route = APIRouter(prefix="/services", tags=["Inferência"])

@services_route.post("/services", tags=["Inferência"])
async def reconhecer_digito(file: UploadFile = File(...)):
    
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail = "Apenas imagens PNG ou JPEG !")

    conteudo = await file.read()
    imagem = carregar_imagem (conteudo)
    imagem_inferir = processa_digito_manual(imagem)

    resultado = modelo.predict(imagem_inferir)
    probabilidade = round(np.max(modelo.predict_proba(imagem_inferir))*100,2)

    return {
        "digito_inferido": resultado,
        "probabilidade": probabilidade
    }



