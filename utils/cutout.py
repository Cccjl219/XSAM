import numpy as np

class Cutout(object):
    def __init__(self, size, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        h, w = img.shape[1:3]

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        img[:, y1:y2, x1:x2] = 0
        return img
