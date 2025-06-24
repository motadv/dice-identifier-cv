import cv2
import numpy as np
import os

def rectify_image(imagePath, output_dir_base='output', output_width=800, output_height=600):
    """
    Retifica uma imagem usando 4 marcadores ArUco e salva os resultados em arquivos
    em vez de exibi-los.

    Args:
        imagePath (str): Caminho para a imagem de entrada.
        output_dir_base (str): Diretório base onde as pastas de saída serão criadas.
        output_width (int): Largura da imagem de saída retificada.
        output_height (int): Altura da imagem de saída retificada.

    Returns:
        tuple: Um par de strings (caminho_antes, caminho_depois) em caso de sucesso,
               ou (caminho_antes, None) em caso de falha na retificação,
               ou (None, None) se nenhum marcador for encontrado.
    """
    # 1. Preparar caminhos e diretórios de saída
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    base_name = os.path.splitext(os.path.basename(imagePath))[0]
    specific_output_dir = os.path.join(output_dir_base, base_name)
    os.makedirs(specific_output_dir, exist_ok=True)

    before_path = os.path.join(specific_output_dir, 'antes.png')
    after_path = os.path.join(specific_output_dir, 'depois.png')

    # 2. Carregar a imagem e detectar marcadores
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Imagem em {imagePath} não pôde ser carregada.")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    corners, ids, rejected = aruco_detector.detectMarkers(image)

    # 3. Processar os resultados
    if ids is not None:
        image_with_markers = image.copy()
        cv2.aruco.drawDetectedMarkers(image_with_markers, corners, ids)
        print(f"DIAGNÓSTICO: Marcadores detectados: {ids.flatten()}")

        if len(ids) == 4 and all(i in ids for i in [0, 1, 2, 3]):
            print("INFO: Todos os 4 marcadores necessários foram encontrados. Prosseguindo para a retificação.")
            
            # --- Lógica de ordenação robusta (sem alterações) ---
            marker_centers = []
            marker_corners_map = {}
            for i, corner_set in zip(ids.flatten(), corners):
                marker_corners_map[i] = corner_set
                cx = np.mean(corner_set[0, :, 0])
                cy = np.mean(corner_set[0, :, 1])
                marker_centers.append((i, (cx, cy)))
            marker_centers.sort(key=lambda p: p[1][1])
            top_markers = sorted(marker_centers[:2], key=lambda p: p[1][0])
            bottom_markers = sorted(marker_centers[2:], key=lambda p: p[1][0])
            tl_id, tr_id = top_markers[0][0], top_markers[1][0]
            bl_id, br_id = bottom_markers[0][0], bottom_markers[1][0]
            print(f"INFO: Ordem dos marcadores detectada -> TL:{tl_id}, TR:{tr_id}, BR:{br_id}, BL:{bl_id}")
            tl_corner = marker_corners_map[tl_id][0][0]
            tr_corner = marker_corners_map[tr_id][0][1]
            br_corner = marker_corners_map[br_id][0][2]
            bl_corner = marker_corners_map[bl_id][0][3]
            src_pts = np.array([tl_corner, tr_corner, br_corner, bl_corner], dtype="float32")
            # --- Fim da lógica ---

            # Desenha o polígono de debug
            pts_for_drawing = np.int32(src_pts).reshape((-1, 1, 2))
            cv2.polylines(image_with_markers, [pts_for_drawing], isClosed=True, color=(0, 255, 255), thickness=4)

            # Salva a imagem 'antes'
            cv2.imwrite(before_path, image_with_markers)
            
            # Realiza a retificação
            dst_pts = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype="float32")
            transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_image = cv2.warpPerspective(image, transform_matrix, (output_width, output_height))
            
            # Salva a imagem 'depois'
            cv2.imwrite(after_path, warped_image)

            print(f"SUCESSO: Imagens salvas em '{specific_output_dir}'")
            return before_path, after_path

        else:
            # Falha na retificação, mas salva a imagem de debug
            print(f"ERRO: Não foi possível detectar todos os 4 marcadores necessários. Salvando imagem de debug.")
            cv2.imwrite(before_path, image_with_markers)
            return before_path, None
    else:
        print("DIAGNÓSTICO: Nenhum marcador foi detectado na imagem.")
        return None, None


if __name__ == "__main__":
    # Iterar o diretíorio de entrada para encontrar imagens
    input_dir = 'input'
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Diretório de entrada '{input_dir}' não encontrado.")
    if not os.listdir(input_dir):
        raise ValueError(f"Diretório de entrada '{input_dir}' está vazio.")

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError(f"Nenhuma imagem encontrada no diretório '{input_dir}'.")

    for image_file in image_files:
        try:
            path_antes, path_depois = rectify_image(os.path.join(input_dir, image_file))
            if path_depois:
                print(f"\nProcessamento concluído.")
                print(f"Imagem de diagnóstico salva em: {path_antes}")
                print(f"Imagem retificada salva em:   {path_depois}")
            elif path_antes:
                print(f"\nProcessamento falhou, mas uma imagem de diagnóstico foi salva em: {path_antes}")
            else:
                print(f"\nProcessamento falhou. Nenhum marcador encontrado em '{image_file}'.")

        except ValueError as e:
            print(f"ERRO CRÍTICO: {e}")