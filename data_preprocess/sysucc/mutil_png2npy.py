import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    out_dir = "/home/xys/NPCNet/data/multimodal"
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/labels', exist_ok=True)

    data_dir = "/home/xys/nnUNet/nnUNet_raw/Dataset002_CT_MRI"

    for label_dir in tqdm(glob(f"{data_dir}/labelsTr/*")):
        label = np.array(Image.open(label_dir), dtype=np.uint8)  # 读取 PNG 文件
        ct_img = np.array(Image.open(label_dir.replace(".png", "_0000.png").replace("labelsTr", "imagesTr")),
                          dtype=np.float32)
        mri_t1_img = np.array(Image.open(label_dir.replace(".png", "_0001.png").replace("labelsTr", "imagesTr")),
                              dtype=np.float32)
        mri_t2_img = np.array(Image.open(label_dir.replace(".png", "_0002.png").replace("labelsTr", "imagesTr")),
                              dtype=np.float32)
        image = np.stack([mri_t1_img, mri_t2_img, ct_img], axis=2)
        np.save(f"{out_dir}/images/{Path(label_dir).stem}.npy", image)
        np.save(f"{out_dir}/labels/{Path(label_dir).stem}.npy", label)
