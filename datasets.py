from torch.utils.data import Dataset
from glob import glob
import numpy as np
from PIL import Image

class PowerlinesDataset(Dataset):
    'Custom primitive Dataset class for powerlines detection task'
    ''''''
    def __init__(self ,train=True):
        super().__init__()
        self.train = train
        self.data = glob(f"data/images/{'train' if train else 'val'}/*")
        self.labels = [i.replace('images', 'labels').replace('jpeg', 'txt') for i in self.data]
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")
        img = np.asarray(img).astype(np.float32)
        img /= 255.
        labels = np.loadtxt(self.labels[idx])
        c, x, y, w, h = labels
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        x1 *= img.shape[1]
        x2 *= img.shape[1]
        y1 *= img.shape[0]
        y2 *= img.shape[0] 
        
        return img.transpose(2, 0, 1), np.array([c, x1, y1, x2, y2]).astype(np.float32)
    