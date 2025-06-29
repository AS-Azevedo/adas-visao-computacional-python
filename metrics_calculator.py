# metrics_calculator.py

import numpy as np

# --- CONSTANTES DE CALIBRAÇÃO ---
# Estes valores são cruciais e precisam de ser ajustados para a sua câmera/vídeo.
# Uma boa calibração é a chave para uma boa estimativa.
# Lembre-se da fórmula: Foco = (Largura_pixels * Distância_conhecida) / Largura_real
FOCAL_LENGTH = 800  # Valor de exemplo
AVG_CAR_WIDTH_REAL = 1.8  # Largura média de um carro em metros

def estimate_distance(box_width_pixels):
    """
    Estima a distância de um objeto com base na largura da sua caixa delimitadora em pixels.
    """
    if box_width_pixels == 0:
        return -1
    
    # Distância = (Largura Real * Distância Focal) / Largura em Pixels
    try:
        distance = (AVG_CAR_WIDTH_REAL * FOCAL_LENGTH) / box_width_pixels
        return distance
    except ZeroDivisionError:
        return -1

def calculate_speed_mps(dist_m_prev, dist_m_curr, time_s):
    """
    Calcula a velocidade em metros por segundo.
    Retorna a velocidade do nosso ponto de vista (negativa se o objeto se aproxima).
    """
    if time_s == 0:
        return 0.0
    
    # Velocidade = (distância final - distância inicial) / tempo
    # Se a distância diminui, a velocidade é negativa (aproximando-se).
    speed_mps = (dist_m_curr - dist_m_prev) / time_s
    return speed_mps