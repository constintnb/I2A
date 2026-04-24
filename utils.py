import numpy as np
import cv2
from PIL import Image

def extract_canny(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img, 150, 250)
    return Image.fromarray(edges)