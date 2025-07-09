# Identificador de Dados por Visão Computacional

Este projeto implementa um pipeline de processamento de imagens para detectar objetos específicos usando um modelo treinado com **YOLOv8** e, em seguida, aplicar Reconhecimento Óptico de Caracteres (OCR) com **EasyOCR** para ler números (de 1 a 20) nos dados D4, D6, D8, D10, D12, D20 dentro das áreas detectadas.

## Autores

  - Rodrigo da Mota
  - Rafael Kanazawa

-----

## 💡 Ideia Geral de Funcionamento

O pipeline opera em duas fases principais:

1.  **Detecção de Objetos**: O script `run_pipeline_ocr.py` carrega uma imagem de entrada e um modelo YOLOv8 treinado. Ele identifica e recorta todas as instâncias de objetos na imagem que atendem a um limiar de confiança especificado.

2.  **Reconhecimento de Caracteres (OCR)**: Para cada objeto recortado (ROI - *Region of Interest*), o pipeline realiza os seguintes passos:

      * Converte a imagem para tons de cinza e a binariza para destacar os números.
      * Para lidar com a orientação dos objetos, a imagem binarizada é rotacionada em incrementos de 90 graus (0°, 90°, 180°, 270°).
      * O EasyOCR é executado em cada uma dessas rotações para ler os números.
      * O número detectado com a maior pontuação de confiança entre todas as rotações é selecionado como o resultado final.
      * As imagens processadas e os resultados são salvos em um diretório de saída.

-----

## ⚙️ Instalação

Para rodar este projeto, clone o repositório e instale as dependências necessárias. É recomendado o uso de um ambiente virtual (`venv` ou `conda`).

1.  **Crie e ative um ambiente virtual (opcional, mas recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

2.  **Instale os pacotes Python:**

    ```bash
    pip install ultralytics easyocr opencv-python pillow numpy
    ```

      * `ultralytics`: Para o modelo YOLOv8.
      * `easyocr`: Para o reconhecimento de caracteres.
      * `opencv-python`, `pillow`, `numpy`: Para manipulação de imagens.

-----

## ▶️ Como Rodar

Execute o pipeline principal através do arquivo `run_pipeline_ocr.py` a partir do seu terminal. Forneça os argumentos necessários para indicar o caminho do modelo, da imagem e outras configurações.

### Comando Básico

Use o seguinte comando para executar o script:

```bash
python run_pipeline_ocr.py --model /caminho/para/seu/modelo.pt --source /caminho/para/sua/imagem.jpg
```

### Argumentos

O script aceita os seguintes argumentos de linha de comando:

| Argumento      | Obrigatório | Descrição                                                                         | Padrão              |
| -------------- | ----------- | ----------------------------------------------------------------------------------- | ------------------- |
| `--model`      | Não | Caminho para o arquivo do modelo YOLOv8 treinado (ex: `best.pt`).                   | `Detector/best.pt`                |
| `--source`     | **Sim** | Caminho para a imagem de entrada que será processada.                               | N/A                 |
| `--confidence` | Não         | Limiar de confiança para a detecção de objetos. Valores entre 0.0 e 1.0.            | `0.5`               |
| `--output_dir` | Não         | Diretório principal onde as saídas e imagens processadas serão salvas.              | `pipeline_output`   |
| `--debug`      | Não         | Se esta flag for usada, salva todas as etapas intermediárias do processamento (ROI, imagem binarizada, etc.). | N/A                 |

### Exemplo de Uso com Debug

Para processar uma imagem chamada `teste.png` com um modelo `yolo_model.pt`, usando uma confiança de `0.7` e salvando todas as etapas intermediárias:

```bash
python run_pipeline_ocr.py --source teste.png --confidence 0.7 --debug
```

Após a execução, um diretório chamado `pipeline_output` (ou o nome que você especificou) será criado, contendo subdiretórios para cada objeto detectado com as respectivas imagens e resultados do OCR.
