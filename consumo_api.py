import os
import requests
import sys
import mimetypes

URL_BASE = "http://127.0.0.1:8000/services"
URL_HEALTH = f"{URL_BASE}/health"
URL_PREDICAO = f"{URL_BASE}/inferencia"

def testar_integridade_api():
    print ("::::: Checagem saúde API (GET/health) :::::")
    print (f"Enviando requisição para: {URL_HEALTH}")

    try:
        response = requests.get (URL_HEALTH)

        if response.status_code == 200:
            print (f"[SUCESSO] Status HTTP: {response.status_code}")
            print (f"-> Resposta da API: {response.json()}\n")
        else:
            print (f"[ERRO] O health check falhou com status {response.status_code}")
            print (f"-> Detalhes: {response.text}")
        
    except requests.exceptions.ConnectionError:
        print("[ERRO DE CONEXÃO] Não foi possível conectar na API.\n")

def executar_inferencia(caminho_imagem: str):
    print ("\n::::: Inferência Preditiva (POST/inferencia) :::::")
    print (f"Arquivo alvo: {caminho_imagem}")
    print(f"Enviando a imagem para: {URL_PREDICAO}")

    try:
        tipo_mime, _ = mimetypes.guess_type(caminho_imagem)
        
        with open(caminho_imagem, "rb") as arquivo_imagem:
            """Essa linha serve para formatar e envelopar o seu arquivo de 
            imagem dentro do padrão exigido pelo protocolo HTTP para uploads,
             chamado multipart/form-data."""
            
            arquivos_requisicao = {
                "file": (os.path.basename(caminho_imagem), arquivo_imagem, tipo_mime)
            }
            response = requests.post(URL_PREDICAO, files=arquivos_requisicao)

        if response.status_code == 200:
            dados = response.json()
            print (f"[SUCESSO] Status HTTP: {response.status_code}")
            print ("::::: RESULTADO DA INFERÊNCIA ::::")
            print (f"Dígito identificado: {dados.get('digito_inferido')}")
            print (f"Probabilidade / certeza: {dados.get('probabilidade')}%")
            print ("-------------------------------------\n")

        else:
            print (f"[ERRO] Status HTTP: {response.status_code}")
            print (f"-> Detalhes: {response.text}")

    except requests.exceptions.ConnectionError:
        print("[ERRO DE CONEXÃO] Não foi possível conectar na API.\n")

if __name__ == "__main__":
    # Verifica a quantidade de argumentos passados via terminal
    # sys.argv[0] é sempre o próprio nome do script (consumo_api.py)
    if len(sys.argv) < 2:
        # Sem parâmetros: Executa apenas o Health Check
        testar_integridade_api()
    else:
        # Com parâmetros: Captura o primeiro argumento e roda a inferência
        imagem_escolhida = sys.argv[1]
        executar_inferencia(imagem_escolhida)