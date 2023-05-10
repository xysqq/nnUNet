import SimpleITK as sitk
import cv2
import itk


def register_images(fixed_volumes, moving_volumes, parameterMap='bspline',
                    rigid_param_file='../RegistrationParameters/Parameters_Rigid.txt',
                    bspline_param_file='../RegistrationParameters/Parameters_BSpline.txt'):
    if isinstance(moving_volumes, list):
        result_arrays = []
        result_transform_parameters = []
        for signal_moving_volumes in moving_volumes:
            result_array, result_transform_parameters = _register_images(fixed_volumes, signal_moving_volumes,
                                                                         parameterMap, rigid_param_file,
                                                                         bspline_param_file)
            result_arrays.append(result_array)
        return result_arrays, result_transform_parameters
    else:
        result_array, result_transform_parameters = _register_images(fixed_volumes, moving_volumes, parameterMap,
                                                                     rigid_param_file, bspline_param_file)
        return result_array, result_transform_parameters


def _register_images(fixed_volumes, moving_volumes, parameterMap,
                     rigid_param_file,
                     bspline_param_file):
    # 初始化刚性配准的参数对象
    parameter_object = itk.ParameterObject.New()

    if parameterMap == 'rigid':
        default_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
        parameter_object.AddParameterMap(default_parameter_map)
    elif parameterMap == 'affine':
        default_parameter_map = parameter_object.GetDefaultParameterMap('affine', 4)
        default_parameter_map['FinalBSplineInterpolationOrder'] = ['0']
        parameter_object.AddParameterMap(default_parameter_map)
    elif parameterMap == 'bspline':
        # parameter_object = update_parameter_map(parameter_object, 'rigid', rigid_param_file)
        # parameter_object = update_parameter_map(parameter_object, 'bspline', bspline_param_file)
        parameter_object.ReadParameterFile(rigid_param_file)
        parameter_object.ReadParameterFile(bspline_param_file)

    moving_volumes = cv2.resize(
        moving_volumes, (fixed_volumes.shape[0], fixed_volumes.shape[1]),
        interpolation=cv2.INTER_CUBIC)

    # 将图像转换为ITK格式
    fixed_image = itk.image_view_from_array(fixed_volumes)
    moving_image = itk.image_view_from_array(moving_volumes)

    # 执行配准算法
    registered_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=True)

    # 将 ITK 图像转换为 SimpleITK 图像
    sitk_image = sitk.GetImageFromArray(registered_image)

    # 将 SimpleITK 图像转换为 NumPy 数组
    registered_array = sitk.GetArrayFromImage(sitk_image)

    return registered_array, result_transform_parameters


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
    dic = read_params_from_file('../RegistrationParameters/Par0044Affine.txt')
