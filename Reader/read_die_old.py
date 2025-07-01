import cv2
import pytesseract
import argparse
import os
import numpy as np

# ==============================================================================
# NOVA FUNÇÃO AUXILIAR PARA CORRIGIR A ROTAÇÃO
# ==============================================================================
def correct_rotation(image):
    """
    Encontra o contorno principal em uma imagem binária e corrige sua rotação
    para deixá-lo na vertical.
    """
    # Adiciona uma borda para evitar que o contorno toque as extremidades
    padded = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image # Retorna a imagem original se nenhum contorno for encontrado

    largest_contour = max(contours, key=cv2.contourArea)

    # Se a área do contorno for muito pequena, ignora para evitar ruído
    if cv2.contourArea(largest_contour) < 50:
        return image

    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    # Ajusta o ângulo para que o texto fique o mais vertical possível
    if rect[1][0] < rect[1][1]: # Se a largura < altura
        angle = angle + 90
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Gira a imagem original (antes de adicionar a borda)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return rotated

def find_top_number(image_path):
    """
    Analisa a imagem de um único dado, corrige a rotação do número
    e tenta identificá-lo com Tesseract.
    """
    # if os.name == 'nt':
    #     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    image = cv2.imread(image_path)
    if image is None:
        return None

    # --- Pré-processamento (como antes) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # --- NOVO PASSO: CORREÇÃO DE ROTAÇÃO ---
    deskewed_binary = correct_rotation(binary)
    
    cv2.imwrite('debug_rotated.png', deskewed_binary)
    
    # --- OCR (agora na imagem corrigida) ---
    config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
    # Mudamos para --psm 10 para tratar a imagem como um único caractere,
    # já que agora ela está centralizada e alinhada.
    
    text = pytesseract.image_to_string(deskewed_binary, config=config)
    
    # A lógica de encontrar o "mais alto" não é mais necessária,
    # pois esperamos ter apenas um número bem centralizado após a rotação.
    result = text.strip()
    
    return result if result.isdigit() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lê o número em uma imagem de um dado, corrigindo a rotação.")
    parser.add_argument("-i", "--image", required=True, help="Caminho para a imagem do dado recortado.")
    args = vars(parser.parse_args())

    result = find_top_number(args["image"])

    if result:
        print(f"\n✅ O resultado do dado é: {result}")
    else:
        print(f"\n❌ Não foi possível determinar o número do dado.")
    
    print("ℹ️ Uma imagem de debug 'debug_rotated.png' foi salva para verificação.")