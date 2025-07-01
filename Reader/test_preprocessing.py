import cv2
import numpy as np
import argparse
import os

def generate_variations(image_path, num_variations):
    """
    Gera N variações de pré-processamento de uma imagem de dado para
    encontrar a melhor combinação para OCR.
    """
    # --- Setup ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível ler a imagem em: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    output_dir = "debug_variations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Gerando {num_variations} variações de pré-processamento. Resultados em '{output_dir}/'")

    # --- Definição dos Parâmetros de Controle ---
    param_profiles = [
        {'blockSize': 11, 'C': 2, 'morph_op': None, 'blur': 3},
        {'blockSize': 25, 'C': 5, 'morph_op': None, 'blur': 3},
        {'blockSize': 15, 'C': 4, 'morph_op': cv2.MORPH_CLOSE, 'blur': 3},
        {'blockSize': 35, 'C': 7, 'morph_op': cv2.MORPH_CLOSE, 'blur': 5},
        {'blockSize': 21, 'C': 10, 'morph_op': cv2.MORPH_OPEN, 'blur': 3},
        {'blockSize': 19, 'C': 5, 'morph_op': cv2.MORPH_CLOSE, 'blur': 0},
    ]

    if num_variations > len(param_profiles):
        print(f"Aviso: O número máximo de perfis pré-definidos é {len(param_profiles)}. Gerando {len(param_profiles)} imagens.")
        num_variations = len(param_profiles)

    # --- Loop de Geração ---
    for i in range(num_variations):
        params = param_profiles[i]
        
        if params['blur'] > 0:
            processed = cv2.medianBlur(gray, params['blur'])
        else:
            processed = cv2.GaussianBlur(gray, (5, 5), 0)

        binary = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, params['blockSize'], params['C']
        )

        if params['morph_op'] is not None:
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, params['morph_op'], kernel)

        filename = f"var_{i+1}_block_{params['blockSize']}_C_{params['C']}.png"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, binary)
        
        print(f"  -> Salvo: {filename}")

# --- BLOCO DE EXECUÇÃO PRINCIPAL (A PARTE QUE FALTAVA) ---
if __name__ == "__main__":
    # Configura o script para aceitar argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description="Gera N variações de pré-processamento de uma imagem de dado para testes."
    )
    parser.add_argument(
        "-i", "--image", 
        required=True, 
        help="Caminho para a imagem do dado recortado."
    )
    parser.add_argument(
        "-n", "--variations", 
        type=int, 
        default=5, 
        help="Número de variações a serem geradas (padrão: 5)."
    )
    args = vars(parser.parse_args())

    # Chama a função principal com os argumentos fornecidos pelo usuário
    generate_variations(image_path=args["image"], num_variations=args["variations"])