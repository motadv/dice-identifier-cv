import cv2
import argparse
import os

def create_canny_output(image_path, output_path):
    """
    Carrega uma imagem, a escala, aplica pré-processamento e salva a saída
    do detector de bordas Canny.

    Args:
        image_path (str): Caminho para a imagem de entrada.
        output_path (str): Caminho onde a imagem Canny de saída será salva.
    """
    print(f"INFO: Processando a imagem: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERRO: Não foi possível ler a imagem em: {image_path}")
        return False

    # 1. Escalar a imagem para ter mais detalhes para trabalhar
    # Usamos INTER_CUBIC para um upscale de alta qualidade.
    scaled_image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 2. Converter para escala de cinza
    gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)

    # 3. Redução de ruído que preserva bordas
    # O filtro bilateral é excelente antes do Canny.
    # Os parâmetros (d, sigmaColor, sigmaSpace) podem ser ajustados.
    # d: Diâmetro da vizinhança de pixels.
    # sigmaColor: Filtra pixels com cores similares.
    # sigmaSpace: Filtra pixels espacialmente próximos.
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=20, sigmaSpace=20)

    # 4. Detecção de bordas com Canny
    # Os dois últimos parâmetros são os limiares inferior e superior.
    # Ajustá-los controla o quão sensível a detecção de bordas será.
    canny_output = cv2.Canny(blurred, 30, 120)

    # 5. Salvar a imagem final
    try:
        cv2.imwrite(output_path, canny_output)
        print(f"SUCESSO: Imagem de bordas Canny salva em: {output_path}")
        return True
    except Exception as e:
        print(f"ERRO: Não foi possível salvar a imagem em {output_path}. Causa: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera uma imagem de bordas Canny a partir de uma imagem de entrada.")
    
    parser.add_argument(
        "-i", "--image", 
        required=True, 
        help="Caminho para a imagem de entrada (ex: d12.png)."
    )
    parser.add_argument(
        "-o", "--output", 
        default="canny_result.png", 
        help="Nome do arquivo de saída (padrão: canny_result.png)."
    )
    args = vars(parser.parse_args())

    create_canny_output(args["image"], args["output"])