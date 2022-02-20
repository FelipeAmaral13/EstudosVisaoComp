
def Dist_Focal(distancia_camera: int, tamanho_rosto: int, tam_rosto_cam: float )-> float:
    """
    Funcao para calcular a distancia focal da camera e o rosto
        distancia_camera: É a distância medida do objeto até a câmera durante a captura da imagem de referência
        tamanho_rosto: Tamanho do rosto de referencia. Variavel face_tam
        tam_rosto_cam: Tamanho da caixa delimitadora calculada quando detectado o rosto com Haarcascade

        saida:
             distancia focal
    """

    distancia_focal = (tam_rosto_cam * distancia_camera)/tamanho_rosto

    return distancia_focal


def Dista_Medida(tamanho_rosto: int, dist_focal_calculada: float, tam_rosto_cam: int):

    # Esta função simplesmente estima a distância entre o objeto e a câmera usando argumentos 
    # (dist_focal_calculada, dist_focal_calculada, tam_rosto_cam)

   distance = (tamanho_rosto * dist_focal_calculada)/tam_rosto_cam 

   return distance
