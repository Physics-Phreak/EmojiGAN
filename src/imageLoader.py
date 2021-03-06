import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

import os
import dotenv
class ImageLoader(Dataset):

    def __init__(self, transform=None):

        dotenv.load_dotenv()
        self.path = os.getenv("IMG_PATH")
        self.imgList = os.listdir(self.path)

        self.transform = transform

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        img = self.processImg(index)
        img = img.astype(np.float32)

        if self.transform:
            img = self.transform(img)

        return img

    def processImg(self, index):
        img = cv2.imread(self.path + "/" + self.imgList[index], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        return img

    def getImgList(self):
        return self.imgList

    def getPath(self):
        return self.path

if __name__=="__main__":
    loader = ImageLoader()
    print(loader.__getitem__(10))