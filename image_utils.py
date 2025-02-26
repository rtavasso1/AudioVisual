# image_utils.py

import cv2
import numpy as np

class ImageUtils:
    @staticmethod
    def resize_and_crop(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = target_width / float(w)
        new_h = int(round(h * scale))
        resized = cv2.resize(image, (target_width, new_h))
        if new_h > target_height:
            start_y = (new_h - target_height) // 2
            cropped = resized[start_y:start_y + target_height, :]
        elif new_h < target_height:
            pad_top = (target_height - new_h) // 2
            pad_bottom = target_height - new_h - pad_top
            cropped = cv2.copyMakeBorder(resized, pad_top, pad_bottom, 0, 0,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            cropped = resized
        return cropped
