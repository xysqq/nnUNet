import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import json
from pathlib import Path
from nnunetv2.paths import nnUNet_results, nnUNet_raw, nnUNet_preprocessed
import argparse
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    arser.add_argument('dataset_name', type=str,
                       help="Dataset name to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('-fold', type=int, default=0,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    args = parser.parse_args()

    if args.configuration == '2d':
        configuration_dir_name = 'nnUNetTrainer__nnUNetPlans__2d'
    elif args.configuration == '3d_fullres':
        configuration_dir_name = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    fold_dir_name = f"fold_{args.fold}"

    splits_file_path = f"{nnUNet_preprocessed}/{args.dataset_name}/splits_final.json"
    with open(splits_file_path, 'r') as f:
        data = json.load(f)
        val_id_list = [image.split('_')[1] for image in data[fold]['val']]

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        str(Path(nnUNet_results, f'{args.dataset_name}/{args.configuration_dir_name}')),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    result = predictor.predict_from_files([[image for image in glob(f"{nnUNet_raw}/{args.dataset_name}/imagesTr/*") if
                                            image.split('_')[1] in val_id_list]],
                                          output_folder_or_list_of_truncated_output_files=None,
                                          save_probabilities=False, overwrite=False,
                                          num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                          folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
