import cv2
import numpy as np
import mediapipe as mp
from skimage.exposure import match_histograms
import argparse

mp_face_mesh = mp.solutions.face_mesh

def get_landmarks(image: np.ndarray, face_mesh: mp_face_mesh.FaceMesh) -> list[tuple[int, int]] | None:
    """
    Detecta os pontos de referência faciais em uma imagem.

    Args:
        image: A imagem de entrada (formato BGR).
        face_mesh: O objeto FaceMesh do MediaPipe.

    Returns:
        Uma lista de tuplas (x, y) dos pontos de referência ou None se nenhum rosto for encontrado.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks:
        return None

    # Assume o primeiro rosto detectado
    landmarks_obj = results.multi_face_landmarks[0]
    
    h, w = image.shape[:2]
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_obj.landmark]
    return landmarks

def delaunay_triangulation(rect: tuple[int, int, int, int], points: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """
    Calcula a triangulação de Delaunay para um conjunto de pontos.
    Versão otimizada usando um dicionário para lookup de índice.
    """

    points_dict = {p: i for i, p in enumerate(points)}
    
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    
    triangle_list = subdiv.getTriangleList()
    triangles = []
    
    for t in triangle_list:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        
        # Verifica se os pontos do triângulo estão dentro do retângulo
        if rect[0] <= pts[0][0] <= rect[2] and rect[1] <= pts[0][1] <= rect[3] and \
           rect[0] <= pts[1][0] <= rect[2] and rect[1] <= pts[1][1] <= rect[3] and \
           rect[0] <= pts[2][0] <= rect[2] and rect[1] <= pts[2][1] <= rect[3]:
            
            indices = [points_dict.get(p) for p in pts]
            
            if all(idx is not None for idx in indices):
                triangles.append(tuple(indices))
                
    return triangles

def warp_triangle(src_img: np.ndarray, dst_img: np.ndarray, t_src: list, t_dst: list):
    """
    Deforma um triângulo da imagem de origem para a imagem de destino.
    """
    r_dst = cv2.boundingRect(np.float32([t_dst]))
    x, y, w, h = r_dst

    # Corta o triângulo de destino para processamento
    t_dst_rect = [(p[0] - x, p[1] - y) for p in t_dst]

    mask = np.zeros((h, w, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_dst_rect), (1.0, 1.0, 1.0), 16, 0)

    # Deforma o triângulo da imagem de origem
    r_src = cv2.boundingRect(np.float32([t_src]))
    src_rect_img = src_img[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]
    t_src_rect = [(p[0] - r_src[0], p[1] - r_src[1]) for p in t_src]

    # Matriz de transformação afim
    M = cv2.getAffineTransform(np.float32(t_src_rect), np.float32(t_dst_rect))
    warped_src_rect = cv2.warpAffine(src_rect_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Combina a área deformada com a imagem de destino
    dst_rect_area = dst_img[y:y+h, x:x+w]
    dst_rect_area = dst_rect_area * (1 - mask) + warped_src_rect * mask
    dst_img[y:y+h, x:x+w] = dst_rect_area

def elliptical_face_mask(size: tuple[int, int], feather: float = 0.20) -> np.ndarray:
    """Cria uma máscara elíptica suavizada para blending."""
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.43), int(h * 0.57))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    k_size = int(min(w, h) * feather)
    if k_size % 2 == 0: k_size += 1
    if k_size < 3: k_size = 3
        
    mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)
    return mask

def create_comparison_view(original: np.ndarray, source: np.ndarray, result: np.ndarray, height: int) -> np.ndarray:
    """Cria uma visualização lado a lado para comparação."""
    def resize_and_label(img, text):
        h, w = img.shape[:2]
        scale = height / h
        resized = cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_AREA)
        cv2.putText(resized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return resized

    original_vis = resize_and_label(original, "ORIGINAL")
    source_vis = resize_and_label(source, "SOURCE")
    result_vis = resize_and_label(result, "RESULT")
    
    return np.hstack([original_vis, source_vis, result_vis])

def draw_landmarks(image: np.ndarray, landmarks: list[tuple[int, int]], color=(0, 255, 255)) -> np.ndarray:
    """Desenha os pontos de referência faciais na imagem."""
    img_copy = image.copy()
    for (x, y) in landmarks:
        cv2.circle(img_copy, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
    return img_copy

def create_comparison_view(original: np.ndarray, source: np.ndarray, result: np.ndarray, height: int,
                          show_landmarks=False, landmarks=None) -> np.ndarray:
    """Cria uma visualização lado a lado para comparação, desenhando landmarks se solicitado."""
    def resize_and_label(img, text):
        h, w = img.shape[:2]
        scale = height / h
        resized = cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_AREA)
        cv2.putText(resized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return resized

    original_vis = original
    if show_landmarks and landmarks is not None:
        original_vis = draw_landmarks(original_vis, landmarks, color=(0,255,255))
    original_vis = resize_and_label(original_vis, "ORIGINAL")
    source_vis = resize_and_label(source, "SOURCE")
    result_vis = resize_and_label(result, "RESULT")
    return np.hstack([original_vis, source_vis, result_vis])


def run_face_swap(source_path: str, camera_id: int):
    """
    Executa o processo de face swap em tempo real a partir de uma câmera.
    """
    src_img = cv2.imread(source_path)
    if src_img is None:
        print(f"Erro: Não foi possível encontrar a imagem de origem em '{source_path}'")
        return

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a câmera com ID {camera_id}")
        return

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh_src, \
         mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh_dst:

        src_landmarks = get_landmarks(src_img, face_mesh_src)
        if src_landmarks is None:
            print("Erro: Nenhum rosto encontrado na imagem de origem.")
            return

        h, w = src_img.shape[:2]
        rect = (0, 0, w, h)
        triangles = delaunay_triangulation(rect, src_landmarks)

        display_height = 480
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            dst_landmarks = get_landmarks(frame, face_mesh_dst)
            output = frame.copy()

            if dst_landmarks:
                # 1. Warping Delaunay
                for tri_indices in triangles:
                    t_src = [src_landmarks[i] for i in tri_indices]
                    t_dst = [dst_landmarks[i] for i in tri_indices]
                    warp_triangle(src_img, output, t_src, t_dst)
                
                # 2. Correção de Cor e Blending
                hull_indices = cv2.convexHull(np.array(dst_landmarks), returnPoints=False)
                hull_points = np.array([dst_landmarks[i[0]] for i in hull_indices])
                
                bbox = cv2.boundingRect(hull_points)
                x, y, w_box, h_box = bbox
                
                # Garante que a ROI não esteja vazia
                if w_box > 0 and h_box > 0:
                    face_dst_roi = frame[y:y+h_box, x:x+w_box]
                    face_src_roi = output[y:y+h_box, x:x+w_box]
                    
                    try:
                        # Correção de cor
                        face_src_roi = match_histograms(face_src_roi, face_dst_roi, channel_axis=-1).astype(np.uint8)
                        output[y:y+h_box, x:x+w_box] = face_src_roi
                    except ValueError:
                        # Ocorre se a ROI for muito pequena ou de cor sólida
                        pass

                    # 3. Blending final com máscara
                    face_mask = elliptical_face_mask((h_box, w_box), feather=0.18)
                    mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)
                    mask_total[y:y+h_box, x:x+w_box] = face_mask

                    center = (x + w_box // 2, y + h_box // 2)
                    try:
                        # Usa seamlessClone para um blending suave
                        output = cv2.seamlessClone(output, frame, mask_total, center, cv2.MIXED_CLONE)
                    except cv2.error:
                        # Ocorre se o centro estiver muito perto da borda
                        pass

            combined_view = create_comparison_view(
                frame, src_img, output, display_height,
                show_landmarks=args.show_landmarks,
                landmarks=dst_landmarks if args.show_landmarks else None
            )

            cv2.imshow("Face Swap Comparativo", combined_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realiza troca de rostos em tempo real usando MediaPipe.")
    parser.add_argument("--source", type=str, required=True, help="Caminho para a imagem do rosto de origem.")
    parser.add_argument("--camera", type=int, default=0, help="ID da câmera a ser usada (padrão: 0).")
    parser.add_argument("--show-landmarks", action='store_true', help="Se ativo, exibe landmarks faciais na imagem original.")

    args = parser.parse_args()
    
    run_face_swap(args.source, args.camera)