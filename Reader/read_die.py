import cv2
import pytesseract
import argparse
import os
import numpy as np
import shutil  # <-- NOVO: Importa a biblioteca para manipulação de diretórios

# --- MÓDULO 1: PRÉ-PROCESSAMENTO (NOVA PIPELINE SIMPLIFICADA E ROBUSTA) ---
def preprocess_image(image_path, debug_folder_path, base_filename):
    """
    Implementa uma pipeline focada em obter um contorno de dado confiável
    para criar uma máscara e isolar os números.
    """
    image = cv2.imread(image_path)
    if image is None: return None, None

    scaled_image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    
    # --- FASE 1: OBTER A MÁSCARA DO DADO ---

    # 1.1. Desfoque que preserva bordas
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    cv2.imwrite(os.path.join(debug_folder_path, f"{base_filename}_1_blurred.png"), blurred)
    
    # 1.2. Detecção de bordas com Canny
    canny = cv2.Canny(blurred, 50, 100)
    cv2.imwrite(os.path.join(debug_folder_path, f"{base_filename}_2_canny_edges.png"), canny)
    
    # 1.3. "Solda" para fechar o contorno do dado. Este é o passo chave.
    # Um kernel grande conecta quebras distantes.
    kernel = np.ones((25, 25), np.uint8)
    closed_canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(debug_folder_path, f"{base_filename}_3_canny_closed.png"), closed_canny)
    
    # 1.4. Encontrar o contorno externo do dado
    contours, _ = cv2.findContours(closed_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("ERRO: Não foi possível encontrar o contorno do dado na Fase 1.")
        return None, None
    die_contour = max(contours, key=cv2.contourArea)
    
    # 1.5. Criar a máscara a partir do contorno confiável
    die_mask = np.zeros_like(gray)
    cv2.drawContours(die_mask, [die_contour], -1, 255, -1)
    cv2.imwrite(os.path.join(debug_folder_path, f"{base_filename}_4_die_mask.png"), die_mask)
    
    # --- FASE 2: EXTRAIR OS NÚMEROS DE DENTRO DA MÁSCARA ---
    
    # 2.1. Aumenta o contraste APENAS na área do dado para realçar os números
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray) # Aplica na 'gray' original para mais detalhes
    isolated_die = cv2.bitwise_and(contrast_enhanced, contrast_enhanced, mask=die_mask)
    cv2.imwrite(os.path.join(debug_folder_path, f"{base_filename}_5_isolated_die.png"), isolated_die)
    
    # 2.2. Encontrar as bordas apenas dos números
    number_edges = cv2.Canny(isolated_die, 100, 200) # Thresholds mais altos aqui funcionam bem
    cv2.imwrite(os.path.join(debug_folder_path, f"{base_filename}_6_number_edges.png"), number_edges)
    
    # 2.3. Preencher os contornos dos números para criar a imagem binária final
    number_contours, _ = cv2.findContours(number_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_binary = np.zeros_like(gray)
    cv2.drawContours(final_binary, number_contours, -1, 255, -1)
    cv2.imwrite(os.path.join(debug_folder_path, f"{base_filename}_7_final_binary.png"), final_binary)
    
    return scaled_image, final_binary

def perform_ocr_rotations(binary_image, num_rotations, debug_folder_path, base_filename):
    """Testa múltiplas rotações na imagem binária e salva cada passo em uma subpasta de debug."""
    (h, w) = binary_image.shape
    center = (w // 2, h // 2)
    all_detections = []
    rotations_debug_path = os.path.join(debug_folder_path, "ocr_rotations")
    os.makedirs(rotations_debug_path, exist_ok=True)
    print(f"INFO: Testando {num_rotations} rotações (debug em '{rotations_debug_path}/')...")
    for i in range(num_rotations):
        angle = i * (360.0 / num_rotations)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_binary = cv2.warpAffine(binary_image, M, (w, h), borderValue=0)
        rotated_debug_vis = cv2.cvtColor(rotated_binary, cv2.COLOR_GRAY2BGR)
        config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789'
        data = pytesseract.image_to_data(rotated_binary, config=config, output_type=pytesseract.Output.DICT)
        inv_M = cv2.invertAffineTransform(M)
        for j in range(len(data['level'])):
            conf = int(data['conf'][j])
            text = data['text'][j].strip()
            if conf > 40 and text.isdigit():
                (x_r, y_r, w_r, h_r) = (data['left'][j], data['top'][j], data['width'][j], data['height'][j])
                cv2.rectangle(rotated_debug_vis, (x_r, y_r), (x_r + w_r, y_r + h_r), (0, 255, 0), 2)
                label = f"{text} ({conf}%)"
                cv2.putText(rotated_debug_vis, label, (x_r, y_r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if conf > 60:
                    rect_r = np.array([[x_r, y_r], [x_r + w_r, y_r + h_r]], dtype="float32")
                    orig_rect = cv2.transform(np.array([rect_r]), inv_M)[0]
                    x_o, y_o = orig_rect[0]; x2_o, y2_o = orig_rect[1]
                    w_o, h_o = x2_o - x_o, y2_o - y_o
                    all_detections.append({'text': text, 'conf': conf, 'box_orig': (int(x_o), int(y_o), int(w_o), int(h_o))})
        debug_filename = f"{base_filename}_rot_{angle:.1f}_deg.png"
        debug_filepath = os.path.join(rotations_debug_path, debug_filename)
        cv2.imwrite(debug_filepath, rotated_debug_vis)
    return all_detections

def filter_detections(detections):
    valid_detections = []
    min_area, max_area = 100, 5000
    min_aspect_ratio, max_aspect_ratio = 0.2, 5.0
    for det in detections:
        conf = det['conf']; x, y, w, h = det['box_orig']
        if conf < 65: continue
        area = w * h
        if not (min_area < area < max_area): continue
        if w > 0 and h > 0:
            aspect_ratio = h / w
            if not (min_aspect_ratio < aspect_ratio < max_aspect_ratio): continue
        else: continue
        valid_detections.append(det)
    return valid_detections

def select_best_detection(detections):
    if not detections: return None
    return min(detections, key=lambda d: d['box_orig'][1])

def create_debug_image(image, valid_detections, top_detection, debug_folder_path, base_filename):
    for det in valid_detections:
        x, y, w, h = det['box_orig']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if top_detection:
        x_top, y_top, w_top, h_top = top_detection['box_orig']
        result_text, result_conf = top_detection['text'], top_detection['conf']
        cv2.rectangle(image, (x_top, y_top), (x_top + w_top, y_top + h_top), (0, 255, 0), 4)
        label = f"Resultado: {result_text} (Conf: {result_conf}%)"
        cv2.putText(image, label, (x_top, y_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    debug_filename = f"{base_filename}_8_final_result.png" # Renomeado para ser o último passo
    output_path = os.path.join(debug_folder_path, debug_filename)
    cv2.imwrite(output_path, image)
    return output_path

# --- FUNÇÃO PRINCIPAL ORQUESTRADORA (COM A MUDANÇA) ---
def find_top_number_orchestrator(image_path, num_rotations=8):
    """Orquestra o pipeline completo, limpando a pasta de debug antes de começar."""
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    debug_folder = f"debug_{base_filename}"

    # ==============================================================================
    # >> NOVO: Limpa o diretório de debug anterior, se existir
    # ==============================================================================
    if os.path.exists(debug_folder):
        print(f"INFO: Limpando diretório de debug anterior: '{debug_folder}/'")
        shutil.rmtree(debug_folder)
    # ==============================================================================

    os.makedirs(debug_folder, exist_ok=True)
    print(f"INFO: Criada nova pasta de debug: '{debug_folder}/'")

    scaled_image, binary_image = preprocess_image(image_path, debug_folder, base_filename)
    if binary_image is None: return None, None

    # all_detections = perform_ocr_rotations(binary_image, num_rotations, debug_folder, base_filename)
    # if not all_detections: return None, debug_folder

    # valid_detections = filter_detections(all_detections)
    # if not valid_detections:
    #     debug_path = create_debug_image(scaled_image, [], None, debug_folder, base_filename)
    #     return None, debug_path

    # top_detection = select_best_detection(valid_detections)
    
    # debug_path = create_debug_image(scaled_image, valid_detections, top_detection, debug_folder, base_filename)

    # if top_detection:
    #     return top_detection['text'], debug_path
    # else:
    #     return None, debug_path

# --- BLOCO DE EXECUÇÃO (sem alterações) ---
if __name__ == "__main__":
    # if os.name == 'nt':
    #     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    parser = argparse.ArgumentParser(description="Lê o número de um dado com teste de múltiplas rotações e debug detalhado.")
    parser.add_argument("-i", "--image", required=True, help="Caminho para a imagem do dado.")
    parser.add_argument("-r", "--rotations", type=int, default=8, help="Número de rotações a serem testadas.")
    args = vars(parser.parse_args())

    find_top_number_orchestrator(args["image"], args["rotations"])

    # result, debug_info = find_top_number_orchestrator(args["image"], args["rotations"])

    # if result:
    #     print(f"\n✅ O resultado do dado é: {result}")
    #     print(f"ℹ️ Imagens de debug salvas em: {os.path.dirname(debug_info)}/")
    # else:
    #     print(f"\n❌ Não foi possível determinar o número do dado após a filtragem.")
    #     if debug_info:
    #         print(f"ℹ️ Verifique a pasta de debug para análise: {debug_info}/")