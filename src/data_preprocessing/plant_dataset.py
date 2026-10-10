from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def get_img_pairs(img_dir, mask_dir):
    img_p, mask_p = [], []
    for img in sorted(img_dir.iterdir()):
        mask = mask_dir / str(img.stem + '.png')
        if mask.exists():
            img_p.append(str(img))
            mask_p.append(str(mask))
        else:
            print(f"No Mask for {img.name}")
    return img_p, mask_p

class PlantDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(384,384)):
        self.img_paths, self.mask_paths = get_img_pairs(img_dir, mask_dir)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = img.resize(self.img_size)
        img_arr = np.array(img, dtype=np.float32) / 255.0

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize(self.img_size, resample=Image.NEAREST)
        mask_arr = np.array(mask, dtype=np.float32)
        mask_arr = (mask_arr > 0).astype(np.float32)
        mask_arr = np.expand_dims(mask_arr, axis=-1)

        img_arr = img_arr.transpose(2 , 0, 1)
        mask_arr = mask_arr.transpose(2, 0, 1)
        return img_arr, mask_arr