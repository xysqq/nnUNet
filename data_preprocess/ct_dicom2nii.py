import logging
import os
import platform
import time
from glob import glob

import mmcv
import numpy as np
import pydicom
from tqdm import tqdm

from utils.utils_dicom import load_rtstruct_file, get_patient_id_from_dicom
from utils.utils_file_conversion import dicom2nifti, img2nifti

if __name__ == '__main__':
    # 创建logger对象
    logger = mmcv.get_logger('ct_dicom2nii')
    # 设置logger的级别
    logger.setLevel(logging.DEBUG)

    # 创建FileHandler对象，设置日志文件的名称和输出格式
    file_handler = logging.FileHandler('../logs/ct_dicom2nii.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将FileHandler对象添加到logger中
    logger.addHandler(file_handler)

    # 原数据集路径
    if platform.system() == 'Windows':
        data_dir = "D:/NPC"
    else:
        data_dir = "/home/xys/Data"
    out_dir = f"../nnUNet_raw/Dataset003_CT_raw"
    os.makedirs(f'{out_dir}/imagesTr', exist_ok=True)
    os.makedirs(f'{out_dir}/labelsTr', exist_ok=True)
    logger.info(f"输出路径为{out_dir}")

    # 读取映射表
    category_map = {
        ' '.join(x.strip().split()[:-1]): x.strip().split(' ')[-1].strip()
        for x in open(rf"{data_dir}/label_target.txt", encoding='utf-8').readlines()
    }
    category_list = ['GTV', 'GTVnd', 'CTV1', 'CTV2']
    logger.info(f"映射表为{category_map}\n列表为{category_list}")
    dataset_json = {"channel_names": {
        "0": "CT",
    }, "file_ending": ".nii.gz", "overwrite_image_reader_writer": "SimpleITKIO",
        'labels': {category: category_idx + 1 for category_idx, category in enumerate(category_list)}}
    dataset_json['labels']['background'] = 0
    ct_dicom_dirs = glob(rf"{data_dir}/npc/*/首次CT/*/*/CT")

    start_time = time.time()
    num_training = 0
    # 遍历每一个患者的文件夹
    for ct_dicom_dir in tqdm(ct_dicom_dirs):
        # 读取数据
        patient_id = get_patient_id_from_dicom(ct_dicom_dir)
        ct_dir = glob(rf"{data_dir}/npc/{patient_id}/首次CT/*/*/CT")[0]
        ct_structure = glob(
            rf"{data_dir}/npc/{patient_id}/首次CT/*/*/RTSTRUCT/*")[0]
        rtstruct, roi_names, body_masks = load_rtstruct_file(ct_dicom_dir, ct_structure)

        # 靶区的mask读取
        target_masks = {}
        for roi_name in roi_names:
            if roi_name in category_map.keys():
                if category_map[roi_name] in target_masks.keys():
                    target_masks[category_map[roi_name]] += rtstruct.get_roi_mask_by_name(roi_name)
                target_masks[category_map[roi_name]] = rtstruct.get_roi_mask_by_name(roi_name)

        # 逐切片处理
        for idx, file_dateset in enumerate(rtstruct.series_data):
            ct_dicom_path = file_dateset.filename
            slice_location = file_dateset.SliceLocation
            if slice_location < -150 or slice_location > 108:
                continue

            # 读取DICOM文件
            dicom_data = pydicom.dcmread(ct_dicom_path)
            image_array = dicom_data.pixel_array
            height, width = image_array.shape[0], image_array.shape[1]
            # 提取spacing和origin
            spacing = [dicom_data.PixelSpacing[0], dicom_data.PixelSpacing[1], dicom_data.SliceThickness]
            origin = [dicom_data.ImagePositionPatient[0], dicom_data.ImagePositionPatient[1],
                      dicom_data.ImagePositionPatient[2]]

            masks = [target_masks[category][..., idx].astype(int) if category in target_masks.keys() else np.zeros(
                (height, width)) for category in category_list]

            class_seg = np.zeros((height, width))
            for i, mask in enumerate(masks):
                class_seg = np.where(
                    class_seg > 0, class_seg, mask * (i + 1))

            dicom2nifti(ct_dicom_path, f"{out_dir}/imagesTr/{patient_id}_{num_training}_0000.nii.gz")
            img2nifti(class_seg, f"{out_dir}/labelsTr/{patient_id}_{num_training}.nii.gz", spacing=spacing,
                      origin=origin)
            num_training += 1

    dataset_json['numTraining'] = num_training
    logger.info(f"dataset.json为{dataset_json}")
    mmcv.dump(dataset_json, f"{out_dir}/dataset.json")

    end_time = time.time()
    print(f"处理数据共用了{(end_time - start_time) / 60:.2f}分钟")
