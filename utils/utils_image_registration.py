import os
import platform
from glob import glob

import SimpleITK as sitk
import cv2
import itk
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils_dicom import dicom2img
from utils.utils_file_ops import remove_and_make_dirs
from utils.utils_image_matching import load_dicom_volumes


def init_registration_parameter_object(parameterMap):
    # 初始化刚性配准的参数对象
    parameter_object = itk.ParameterObject.New()
    # parameters_txt_path = f"../RegistrationParameters/Rigid.txt"
    # parameter_object.AddParameterFile(parameters_txt_path)

    if parameterMap == 'rigid':
        default_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    elif parameterMap == 'affine':
        default_parameter_map = parameter_object.GetDefaultParameterMap('affine', 4)
        default_parameter_map['FinalBSplineInterpolationOrder'] = ['0']

    elif parameterMap == 'bspline':
        default_parameter_map = parameter_object.GetDefaultParameterMap('affine', 4)
        default_parameter_map['FinalBSplineInterpolationOrder'] = ['1']
        parameter_object.AddParameterMap(default_parameter_map)
        default_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 4)
        default_parameter_map['FinalBSplineInterpolationOrder'] = ['1']
    parameter_object.AddParameterMap(default_parameter_map)
    return parameter_object


def ct_and_mri_registered(fixed_image_path, moving_image_path, parameter_object):
    fixed_image = dicom2img(fixed_image_path).astype('float32')
    moving_image = dicom2img(moving_image_path).astype('float32')

    fixed_image[330:, :] = 0

    # 将mri_img分辨率下采样到ct_img图像大小
    moving_image = cv2.resize(
        moving_image, (fixed_image.shape[0], fixed_image.shape[1]),
        interpolation=cv2.INTER_CUBIC)

    # 将图像转换为ITK格式
    fixed_image = itk.image_view_from_array(fixed_image)
    moving_image = itk.image_view_from_array(moving_image)

    # 执行配准算法
    registered_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=True)

    return itk.array_view_from_image(
        registered_image), result_transform_parameters


def ct_and_mri_registered_simple_elastix(fixed_volumes, moving_volumes,
                                         rigid_param_file='../RegistrationParameters/Parameters_Rigid.txt',
                                         bspline_param_file='../RegistrationParameters/Parameters_BSpline.txt'):
    if isinstance(moving_volumes, list):
        result_arrays = []
        for signal_moving_volumes in moving_volumes:
            result_array, [] = _ct_and_mri_registered_simple_elastix(fixed_volumes, signal_moving_volumes,
                                                                     rigid_param_file=rigid_param_file,
                                                                     bspline_param_file=bspline_param_file)
            result_arrays.append(result_array)
        return result_arrays, []
    else:
        result_array, [] = _ct_and_mri_registered_simple_elastix(fixed_volumes, moving_volumes,
                                                                 rigid_param_file=rigid_param_file,
                                                                 bspline_param_file=bspline_param_file)
        return result_array, []


def _ct_and_mri_registered_simple_elastix(fixed_volumes, moving_volumes,
                                          rigid_param_file='../RegistrationParameters/Par0044Affine.txt',
                                          bspline_param_file='../RegistrationParameters/Parameters_BSpline.txt'):
    # 创建和配置 ElastixImageFilter
    elastixImageFilter = sitk.ElastixImageFilter()

    fixed_image = sitk.GetImageFromArray(fixed_volumes)
    elastixImageFilter.SetFixedImage(fixed_image)

    moving_volumes = cv2.resize(
        moving_volumes, (fixed_volumes.shape[0], fixed_volumes.shape[1]),
        interpolation=cv2.INTER_CUBIC)
    moving_image = sitk.GetImageFromArray(moving_volumes)
    elastixImageFilter.SetMovingImage(moving_image)

    # 读取刚性配准参数文件
    rigid_param_map = sitk.ReadParameterFile(rigid_param_file)
    elastixImageFilter.SetParameterMap(rigid_param_map)

    elastixImageFilter.LogToFileOn()
    elastixImageFilter.Execute()

    # 应用B样条配准参数
    bspline_param_map = sitk.ReadParameterFile(bspline_param_file)
    elastixImageFilter.SetParameterMap(bspline_param_map)

    elastixImageFilter.LogToFileOn()
    elastixImageFilter.Execute()

    # 获取结果图像
    result_image = elastixImageFilter.GetResultImage()

    result_array = sitk.GetArrayFromImage(result_image).astype('uint8')

    return result_array, []


if __name__ == '__main__':
    if platform.system() == 'Windows':
        data_dir = "D:/NPC"
    else:
        data_dir = "/home/xys/Data"
    patient_id = '何用胜531033'

    ct_dicom_dir = glob(f"{data_dir}/npc/{patient_id}/首次CT/*/*/CT")[0]
    ct_structure = glob(f"{data_dir}/npc/{patient_id}/首次CT/*/*/RTSTRUCT/*")[0]

    ct_volumes, mri_volumes, ct_dicom_path_list, _ = load_dicom_volumes(ct_dicom_dir,
                                                                     f"{data_dir}/npc/{patient_id}/MR/S2010",
                                                                     ct_structure)
    result_volumes, _ = ct_and_mri_registered_simple_elastix(ct_volumes, mri_volumes)

    # 保存结果图像
    output_dir = f'{data_dir}/elastix_images'
    remove_and_make_dirs(output_dir)

    for idx in tqdm(range(mri_volumes.shape[2])):
        # 绘制 CT 和 MRI 以及结果图像的对比图
        fig, axs = plt.subplots(1, 3, figsize=(30, 30))
        axs[0].imshow(ct_volumes[..., idx], cmap='gray')
        axs[0].set_title('CT')

        axs[1].imshow(mri_volumes[..., idx], cmap='gray')
        axs[1].set_title('MRI')

        axs[2].imshow(result_volumes[..., idx], cmap='gray')
        axs[2].set_title('Registered')

        output_path = os.path.join(output_dir, f'ct_mri_registered_{idx}.png')
        plt.savefig(output_path)
