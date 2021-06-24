import numpy as np

def normalize_open_price(x, open):
    normalized = np.round(np.divide(x, open), 6)
    return normalized