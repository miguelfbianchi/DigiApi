# DigiApi - API de Reconhecimento de Dígitos Manuscritos

## 1. Nome e Resumo do Projeto
O **DigiApi** é um microsserviço focado na classificação automática de dígitos manuscritos isolados (0 a 9). 
A solução disponibiliza um modelo de Machine Learning de forma escalável, transformando um algoritmo experimental em uma ferramenta integrada pronta para produção.

---

## 2. Origem do Modelo
Este projeto reutiliza e aprimora o modelo preditivo desenvolvido anteriormente no módulo de **Classificação de Imagens MNIST / Visão Computacional**. 

O modelo consiste em uma rede perceptron multicamadas (MLP) treinada a partir do dataset público **MNIST** (Modified National Institute of Standards and Technology). Os pesos matemáticos gerados no treinamento offline foram persistidos e serializados no formato de grande objeto binário (BLOB) dentro de uma base de dados relacional local (SQLite), garantindo versionamento e portabilidade do artefato.

---

## 3. Tecnologias Utilizadas
A arquitetura do projeto foi construída utilizando ferramentas baseadas em Python com foco em performance e integridade de dados:

* **FastAPI (v0.115+)**: Framework web assíncrono para construção dos endpoints RESTful.
* **Pydantic (v2.x)**: Validação estrita e tipagem em tempo de execução para os contratos de entrada e saída.
* **OpenCV-Python (cv2)**: Mecanismo de visão computacional responsável pelo pipeline de tratamento e padronização espacial da imagem antes da inferência.
* **Scikit-Learn (v1.5+)**: Framework utilizado para carregar o classificador MLP e computar as predições.
* **Uvicorn (v0.32+)**: Servidor ASGI para sustentação da aplicação web.
* **SciPy & NumPy**: Bibliotecas matemáticas para manipulação de matrizes de pixels e cálculos complexos.

---

## 4. Instruções de Instalação e Execução

Siga os comandos abaixo no seu terminal para clonar, configurar o ambiente e executar o servidor localmente.

### 4.1 Criação do Ambiente Virtual (venv)
No diretório raiz do projeto, execute o comando para isolar as dependências:
```bash
# No Linux/MacOS:
python3 -m venv .venv
source .venv/bin/activate

# No Windows (Prompt de Comando):
python -m venv .venv
.venv\Scripts\activate
```

### 4.2 Instalação das Dependências
Com o seu ambiente `.venv` ativado, instale todas as bibliotecas necessárias listadas no manifesto:
```bash
pip install -r requirements.txt
```

### 4.3 Clonagem do repositório
```bash
git clone git@github.com:miguelfbianchi/DigiApi.git
```

### 4.4 Treinamento e serialização do modelo
Executar o script de treinamento do modelo e gravação do banco de dados local.
```bash
python treinamento_ml.py
```

### 4.5 Inicialização do Servidor com Uvicorn
Para rodar a aplicação com recarregamento automático durante o desenvolvimento, execute:
```bash
uvicorn main:app --reload
```
O servidor estará ativo e respondendo no endereço padrão: `http://127.0.0.1:8000`

---

## 5. Documentação da API

A API foi projetada de forma modular e expõe a documentação interativa oficial gerada automaticamente pelo ecossistema OpenAPI.
* **Link do Swagger UI:** `http://127.0.0.1:8000/docs`

### 5.1 Endpoint de Monitoramento (`GET /services/health`)
Verifica se a aplicação está operacional e se os pesos do modelo de Machine Learning foram carregados corretamente na memória RAM através do ciclo de vida (*lifespan*).

**Exemplo de Resposta (JSON):**
```json
{
  "status": "healthy",
  "modelo_carregado": true
}
```

### 5.2 Endpoint de Inferência (`POST /services/inferencia`)
Consome um arquivo de imagem binária (`PNG` ou `JPEG`) via formulário `multipart/form-data`, submete-o ao pipeline de processamento do OpenCV (suavização, binarização reversa, corte envolvente, centralização por centro de massa e achatamento vetorial) e calcula a inferência.

* **Parâmetro de Entrada (Form):** `file` (Arquivo de Imagem)

**Exemplo de Resposta de Sucesso (HTTP 200 OK):**
```json
{
  "digito_inferido": 3,
  "probabilidade": 100.0
}
```

**Exemplo de Resposta de Erro de Validação (HTTP 422 Unprocessable Entity):**
```json
{
  "detail": "Tipo de arquivo inválido. O corpo da requisição deve conter uma imagem PNG ou JPEG."
}
```

---

## 6. Teste de Consumo da API

Foi escrito um script para testar o consumo da API. Pode ser executado via terminal. Para testar o estado da API e verificar se está ativa e com o modelo de inferência carregado (`GET /services/health`), basta executá-lo sem nenhum parâmetro.

```bash
python consumo_api.py
```

Pode ser executado também informando como parâmetro o nome de um arquivo .jpg ou .png simulando o comportamento da interface do Swagger na rota `POST /services/inferencia`. Faz parte do repositório uma pasta `data/` com alguns arquivos de dígitos manuscritos disponibilizados para o teste.

```bash
python consumo_api.py data/08.jpg
```

---

## 7. Vídeo apresentação

Como parte dos resultados esperados e critérios de avaliação do Projeto Avaliativo, o link abaixo dá acesso à demonstração do ecossistema funcionando de ponta a ponta:

* 🎬 **Vídeo de Demonstração e Defesa Técnica:** [Link do Google Drive Aqui] *(Substitua este texto pelo link em modo leitor quando gerar o vídeo)*
