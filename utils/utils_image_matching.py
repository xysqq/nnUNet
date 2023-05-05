import platform
from glob import glob

import numpy as np
import pydicom

from utils.utils_dicom import ct_dicom2img, load_rtstruct_file, dicom2img, load_mri_dicom_volumes, image_normalize


def match_ct_and_mri(ct_dicom_dir: str, mri_dicom_dir: str) -> dict:
    """
    匹配CT和MRI图像，并返回一个字典，其中键是CT图像的路径，值是与其匹配的MRI图像的路径。
    :param ct_dicom_dir: CT图像的文件夹路径。
    :param mri_dicom_dir: MRI图像的文件夹路径。
    :return: 一个字典，其中键是CT图像的路径，值是与其匹配的MRI图像的路径。
    """
    ct_dicom_path_list = glob(rf"{ct_dicom_dir}/*")
    mri_dicom_path_list = glob(rf"{mri_dicom_dir}/*")

    ct_slice_location = {}
    mri_slice_location = {}

    # 获取CT和MRI图像的位置信息
    for ct_dicom_path in ct_dicom_path_list:
        dcm = pydicom.dcmread(ct_dicom_path)
        slice_location = dcm.get('SliceLocation', None)
        if slice_location is not None:
            ct_slice_location[ct_dicom_path] = float(slice_location)

    for mri_dicom_path in mri_dicom_path_list:
        dcm = pydicom.dcmread(mri_dicom_path)
        slice_location = dcm.get('SliceLocation', None)
        if slice_location is not None:
            # MR的相对位置与CT相反
            mri_slice_location[mri_dicom_path] = -float(slice_location)

    ct_mri_matching_dict = {}
    min_diff = float('inf')
    for ct_key in ct_slice_location:
        matched_key = None
        for mri_key in mri_slice_location:
            # 计算差值
            diff = abs(mri_slice_location[mri_key] - ct_slice_location[ct_key])
            # 如果位置差更小，更新最小位置差和最佳匹配的MRI图像的路径
            if round(diff, 2) <= round(min_diff, 2) and diff < 2:
                min_diff = diff
                matched_key = mri_key
                # 将匹配结果存储在字典中
        if matched_key is not None:
            ct_mri_matching_dict[ct_key] = matched_key
    return ct_mri_matching_dict


def load_dicom_volumes(ct_dicom_dir, mri_dicom_dirs, ct_structure):
    ct_volumes, ct_dicom_path_list, start_index = [], [], None
    if isinstance(mri_dicom_dirs, list):
        total_mri_volumes = []
        for mri_dicom_dir in mri_dicom_dirs:
            ct_volumes, mri_volumes, ct_dicom_path_list, start_index = _load_dicom_volumes(ct_dicom_dir, mri_dicom_dir,
                                                                                           ct_structure)
            total_mri_volumes.append(mri_volumes)
        return ct_volumes, total_mri_volumes, ct_dicom_path_list, start_index
    else:
        ct_volumes, mri_volumes, ct_dicom_path_list, start_index = _load_dicom_volumes(ct_dicom_dir, mri_dicom_dirs,
                                                                                       ct_structure)
        return ct_volumes, mri_volumes, ct_dicom_path_list, start_index


def _load_dicom_volumes(ct_dicom_dir, mri_dicom_dir, ct_structure):
    ct_volumes = []
    mri_volumes = []
    ct_dicom_path_list = []

    ct_mri_matching_dic = match_ct_and_mri(ct_dicom_dir, mri_dicom_dir)
    rtstruct, _, body_masks = load_rtstruct_file(ct_dicom_dir, ct_structure)

    start_index = None
    for idx, file_dateset in enumerate(rtstruct.series_data):
        ct_dicom_path = file_dateset.filename
        if ct_dicom_path not in ct_mri_matching_dic:
            continue
        else:
            start_index = idx if start_index is None else start_index
        ct_volumes.append(ct_dicom2img(ct_dicom_path, body_masks[..., idx]))
        ct_dicom_path_list.append(ct_dicom_path)
        mri_volumes.append(dicom2img(ct_mri_matching_dic[ct_dicom_path], center=512, width=1024))

    return np.stack(ct_volumes, axis=2), np.stack(mri_volumes, axis=2), ct_dicom_path_list, start_index


if __name__ == '__main__':
    if platform.system() == 'Windows':
        data_dir = "D:/NPC"
    else:
        data_dir = "/home/xys/Data"
    patient_id = '关崇步540430'

    ct_dicom_dir = glob(f"{data_dir}/npc/{patient_id}/首次CT/*/*/CT")[0]
    ct_structure = glob(f"{data_dir}/npc/{patient_id}/首次CT/*/*/RTSTRUCT/*")[0]

    ct_volumes, mri_volumes, ct_dicom_path_list, start_index = load_dicom_volumes(ct_dicom_dir,
                                                                                  f"{data_dir}/npc/{patient_id}/MR/S2010",
                                                                                  ct_structure)

    mri_volumes2 = load_mri_dicom_volumes(f"{data_dir}/npc/{patient_id}/MR/S2010")
