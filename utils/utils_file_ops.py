import os
import pathlib
import shutil
import traceback
from pathlib import Path

import mmcv
from mmdet.utils import get_root_logger


def get_roi_names_selected(file_path):
    selected_roi_names = []
    with open(file_path, 'r') as f:
        for line in f:
            words = line.strip().split()
            selected_roi_names.append(' '.join(words[:-1]))
            ' '.join(line.strip().split()[:-1])
        f.close()
    return selected_roi_names


def save_json_data_list(json_data_list, out_dir):
    for i in range(len(json_data_list)):
        json_path = Path(
            out_dir,
            Path(json_data_list[i]['imagePath']).with_suffix('.json'))
        mmcv.dump(json_data_list[i], json_path, indent=2, ensure_ascii=False)


def save_json(json_data_dic, save_path):
    from mmdet.utils import get_root_logger
    logger = get_root_logger()
    logger.info(f'保存路径{save_path}')
    logger.info(f'样本数量：{len(json_data_dic["images"])}')
    logger.info(f'标注shape数量{len(json_data_dic["annotations"])}')
    mmcv.dump(json_data_dic, save_path)


def copy_json_and_img(json_src, target_dir):
    if isinstance(json_src, list):
        for json in json_src:
            json_dst = os.path.join(target_dir, os.path.basename(json))
            _copy_json_and_img(json, json_dst)
    else:
        _copy_json_and_img(json_src, target_dir)


def _copy_json_and_img(json_src, target_dir):
    json_dst = os.path.join(target_dir, os.path.basename(json_src))
    copy_file(json_src, json_dst)
    img_path = mmcv.load(json_src)['imagePath']
    img_dst = os.path.join(target_dir, img_path)
    img_src = str(pathlib.Path(json_src).with_name(img_path))
    copy_file(img_src, img_dst)
    json_data = mmcv.load(json_src)
    if json_data.get('image_path_list', False):
        for image_path in json_data['image_path_list']:
            dirname = Path(json_src).parent
            image_dst = os.path.join(target_dir, image_path)
            image_src = Path(dirname, image_path)
            copy_file(image_src, image_dst)


def copy_file(src, dst):
    logger = get_root_logger()
    src = str(pathlib.Path(src))
    dst = str(pathlib.Path(dst))
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        logger.debug(f"复制 {src} 成功")
    except PermissionError:
        pass
    except:
        logger.error(f"复制 {src} 失败")
        logger.error(traceback.format_exc())


def remove_and_make_dirs(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
