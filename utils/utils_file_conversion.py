import os

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image


def dicom2nifti(dicom_file_path, output_file_path=None):
    # 读取DICOM文件
    dicom_data = pydicom.dcmread(dicom_file_path)

    # 提取图像数据
    image_data = dicom_data.pixel_array

    # 创建NIfTI对象
    nifti_img = nib.Nifti1Image(image_data, None)

    # 更新元数据
    nifti_img.header['pixdim'][1:4] = [dicom_data.PixelSpacing[0], dicom_data.PixelSpacing[1],
                                       dicom_data.SliceThickness]
    nifti_img.header['qoffset_x'] = dicom_data.ImagePositionPatient[0]
    nifti_img.header['qoffset_y'] = dicom_data.ImagePositionPatient[1]
    nifti_img.header['qoffset_z'] = dicom_data.ImagePositionPatient[2]
    nifti_img.header['quatern_b'] = 0.0
    nifti_img.header['quatern_c'] = 0.0
    nifti_img.header['quatern_d'] = 0.0
    nifti_img.header['qform_code'] = 1  # 'scanner'
    nifti_img.header['sform_code'] = 1  # 'scanner'
    nifti_img.header['descrip'] = 'Converted from DICOM to NIfTI format'

    if output_file_path is not None:
        # 保存为NIfTI文件
        nib.save(nifti_img, output_file_path)

    return nifti_img


def img2nifti(img_file, output_file_path=None,  spacing=None, origin=None):
    # 判断img_file是路径还是数组
    if isinstance(img_file, str) and os.path.isfile(img_file):
        # 读取图像文件
        img = Image.open(img_file)
        img_data = np.array(img)
    elif isinstance(img_file, np.ndarray):
        img_data = img_file
    else:
        raise ValueError("Invalid input: img_file should be a file path or a numpy array")

    # 将图像数据转换为nii格式
    nifti_img = nib.Nifti1Image(img_data, np.eye(4))

    # 更新spacing和origin
    if spacing is not None:
        nifti_img.header['pixdim'][1:4] = list(spacing)
    if origin is not None:
        nifti_img.header['qoffset_x'] = origin[0]
        nifti_img.header['qoffset_y'] = origin[1]
        nifti_img.header['qoffset_z'] = origin[2]

    # 保存nii文件到指定路径
    nib.save(nifti_img, output_file_path)

    if output_file_path is not None:
        # 保存为NIfTI文件
        nib.save(nifti_img, output_file_path)

    return nifti_img
