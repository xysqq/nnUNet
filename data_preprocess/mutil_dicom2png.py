import logging
import platform
import time
from glob import glob

import mmcv
import numpy as np
from tqdm import tqdm

from utils.utils_dicom import ct_dicom2img, load_rtstruct_file, get_patient_id_from_dicom
import platform
import time
from glob import glob
from pathlib import Path

import mmcv
import numpy as np
from tqdm import tqdm

from utils.utils_dicom import ct_dicom2img, load_rtstruct_file, get_patient_id_from_dicom
from utils.utils_file_ops import get_roi_names_selected, save_json_data_list, remove_and_make_dirs
from utils.utils_image_matching import load_dicom_volumes, match_ct_and_mri
from utils.utils_image_registration import ct_and_mri_registered_simple_elastix
from utils.utils_labelme import mask2shapes

if __name__ == '__main__':
    # 创建logger对象
    logger = mmcv.get_logger('mutil_dicom2png')
    # 设置logger的级别
    logger.setLevel(logging.DEBUG)

    # 创建FileHandler对象，设置日志文件的名称和输出格式
    file_handler = logging.FileHandler('../logs/mutil_dicom2png.txt')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将FileHandler对象添加到logger中
    logger.addHandler(file_handler)

    # 原数据集路径
    if platform.system() == 'Windows':
        data_dir = "D:/NPC"
    else:
        data_dir = "/home/xys/Data"

    out_dir = f"../nnUNet_raw/Dataset002_CT_MRI"
    logger.info(f"原数据集的路径为{data_dir}/npc，输出路径为{out_dir}")

    # 读取映射表
    category_map = {
        ' '.join(x.strip().split()[:-1]): x.strip().split(' ')[-1].strip()
        for x in open(rf"{data_dir}/label_target.txt", encoding='utf-8').readlines()
    }
    category_list = ['GTV', 'GTVnd', 'CTVnd', 'CTV1', 'CTV2']
    logger.info(f"映射表为{category_map}\n列表为{category_list}")
    dataset_json = {"channel_names": {
        "0": "CT", "1": "MRI_T1", "2": "MRI_T2"
    }, "file_ending": ".png", "overwrite_image_reader_writer": "NaturalImage2DIO",
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
        mri_t1_dir = rf"{data_dir}/npc/{patient_id}/MR/S2010"
        mri_t2_dir = rf"{data_dir}/npc/{patient_id}/MR/S3010"
        mri_dicom_dirs = [mri_t1_dir, mri_t2_dir]
        rtstruct, roi_names, body_masks = load_rtstruct_file(ct_dicom_dir, ct_structure)

        ct_volumes, mri_volumes, ct_dicom_path_list, start_index = load_dicom_volumes(ct_dicom_dir,
                                                                                      mri_dicom_dirs,
                                                                                      ct_structure)
        result_volumes, _ = ct_and_mri_registered_simple_elastix(ct_volumes, mri_volumes)

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

            ct_image = ct_volumes[..., idx - start_index]
            mri_t1_image = result_volumes[0][..., idx - start_index]
            mri_t2_image = result_volumes[1][..., idx - start_index]

            height, width = ct_image.shape[0], ct_image.shape[1]

            masks = [target_masks[category][..., idx].astype(int) if category in target_masks.keys() else np.zeros(
                (height, width)) for category in category_list]

            class_seg = np.zeros((height, width))
            for i, mask in enumerate(masks):
                class_seg = np.where(
                    class_seg > 0, class_seg, mask * (i + 1))

            mmcv.imwrite(ct_image, f"{out_dir}/imagesTr/{patient_id}_{num_training}_0000.png")
            mmcv.imwrite(mri_t1_image, f"{out_dir}/imagesTr/{patient_id}_{num_training}_0001.png")
            mmcv.imwrite(mri_t2_image, f"{out_dir}/imagesTr/{patient_id}_{num_training}_0002.png")
            mmcv.imwrite(class_seg, f"{out_dir}/labelsTr/{patient_id}_{num_training}.png")
            num_training += 1

    dataset_json['numTraining'] = num_training
    logger.info(f"dataset.json为{dataset_json}")
    mmcv.dump(dataset_json, f"{out_dir}/dataset.json")

    end_time = time.time()
    print(f"处理数据共用了{(end_time - start_time) / 60:.2f}分钟")
