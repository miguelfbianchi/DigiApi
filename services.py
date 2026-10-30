#from fastapi import APIRouter, UploadFile, File, HTTPException
import io
import cv2
import numpy as np
from scipy import ndimage
import math
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

def executar_inferencia(modelo, imagem_inferir):
    
    predicao = modelo.predict(imagem_inferir)
    resultado = int(predicao[0])
    probabilidade = round(np.max(modelo.predict_proba(imagem_inferir))*100,2)

    return {
        "digito_inferido": resultado,
        "probabilidade": probabilidade
    }

def carregar_imagem_upload(contents: bytes) -> np.ndarray:
    """Transforma os bytes do upload diretamente na matriz Grayscale que o cv2.imread gerava."""
    matriz_unidimensional = np.frombuffer(contents, dtype=np.uint8)
    # A flag IMREAD_GRAYSCALE garante que a imagem seja decodificada exatamente em tons de cinza e bidimensional
    return cv2.imdecode(matriz_unidimensional, cv2.IMREAD_GRAYSCALE)

def processa_digito_manual(img: np.ndarray):
  
    # Suaviza de leve para tirar imperfeições do papel
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Threshold rígido com inversão (Garante fundo 100% preto de verdade)
    _, thresh = cv2.threshold(img_blur, 130, 255, cv2.THRESH_BINARY_INV)

     # Encontra a caixa envolvente
    contornos, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contornos) == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    maior_contorno = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maior_contorno)
    recorte_digito = thresh[y : y + h, x : x + w]

    # Redimensiona para proporção 20x20
    if w > h:
        novo_w = 20
        novo_h = int(h * (20 / w))
    else:
        novo_h = 20
        novo_w = int(w * (20 / h))

    novo_w, novo_h = max(1, novo_w), max(1, novo_h)
    digito_redimensionado = cv2.resize(
        recorte_digito, (novo_w, novo_h), interpolation=cv2.INTER_AREA
    )

    lona_mnist = np.zeros((28, 28), dtype=np.uint8)
    start_y = (28 - novo_h) // 2
    start_x = (28 - novo_w) // 2
    lona_mnist[start_y : start_y + novo_h, start_x : start_x + novo_w] = (
        digito_redimensionado
    )

    # Centralização por Centro de Massa
    cy, cx = ndimage.center_of_mass(lona_mnist)
    if not math.isnan(cy) and not math.isnan(cx):
        shift_y = int(round(14.0 - cy))
        shift_x = int(round(14.0 - cx))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        lona_mnist = cv2.warpAffine(lona_mnist, M, (28, 28))

     # Engrossa o traço diretamente na lona final de 28x28 para o modelo enxergar
    kernel = np.ones((2, 2), np.uint8)
    lona_mnist = cv2.dilate(lona_mnist, kernel, iterations=1)

    # Aplica um desfoque suave para recriar as bordas esfumaçadas do MNIST original
    lona_mnist = cv2.GaussianBlur(lona_mnist, (3, 3), 0)

    digito_processado = lona_mnist.reshape(784) / 255
    digito_processado = digito_processado.reshape(1, -1)
 
    return digito_processado
