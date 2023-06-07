from glob import glob
from pathlib import Path

import mmcv
from tqdm import tqdm

from data_preprocess.SegRap_2023 import segrap_task001_one_hot_label_names
from utils.utils_file_ops import copy_file

if __name__ == '__main__':
    data_dir = "/home/xys/Data"
    images_dir = f"{data_dir}/SegRap2023_Training_Set_120cases"
    labels_dir = f"{data_dir}/SegRap2023_Training_Set_120cases_OneHot_Labels"
    one_hot_label_names_path = f"{labels_dir}/one_hot_label_names.txt"

    out_dir = "/home/xys/nnUNet/nnUNet_raw/Dataset001_organ_at_risk_CT"
    Path(f"{out_dir}/imagesTr").mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/labelsTr").mkdir(parents=True, exist_ok=True)

    images_path_list = glob(f"{images_dir}/*/image.nii.gz")
    labels_path_list = glob(f"{labels_dir}/Task001/*")
    for image_path, label_path in tqdm(zip(images_path_list, labels_path_list)):
        image_id = f"{Path(image_path).parent.name}_0000.nii.gz"
        label_id = f"{Path(label_path).name.split('.')[0]}.nii.gz"
        copy_file(image_path, f"{out_dir}/imagesTr/{image_id}")
        copy_file(label_path, f"{out_dir}/labelsTr/{label_id}")

    dataset_json = {"channel_names": {
        "0": "CT"
    }, "file_ending": ".nii.gz", "overwrite_image_reader_writer": "SimpleITKIO",
        'labels': {category: category_idx for category, category_idx in segrap_task001_one_hot_label_names.items()},
        'numTraining': 120}
    dataset_json['labels']['background'] = 0
    mmcv.dump(dataset_json, f"{out_dir}/dataset.json")
