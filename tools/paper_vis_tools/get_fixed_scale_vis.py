import os

exp = 'EXP20221219_3'
model_name = 'FunSR-RCAN'  # bicubic, SRCNN, FSRCNN, LGCNet
dataset_name = 'AID'  # UC, AID

for cp in ['epoch-last.pth']:
    for scale_ratio in [4.0]:
        # os.system(f'CUDA_VISIBLE_DEVICES=2 python test_cnn_sr.py '
        #           f'--config tools/paper_tools/vis_fixed_scale_UC_INR_diinn_arbrcan_funsr_overnet.yaml '
        #           f'--model checkpoints/{exp}/{cp} '
        #           f'--scale_ratio {scale_ratio} '
        #           f'--save_fig True '
        #           f'--save_path vis_{model_name}_{dataset_name}_4x_testset '
        #           f'--cal_metrics True '
        #           f'--dataset_name {dataset_name}'
        #           )

        os.system(f'CUDA_VISIBLE_DEVICES=5 python test_inr_diinn_arbrcan_sadnarc_funsr_overnet.py '
                  f'--config tools/paper_tools/vis_fixed_scale_UC_INR_diinn_arbrcan_funsr_overnet.yaml '
                  f'--model checkpoints/{exp}/{cp} '
                  f'--scale_ratio {scale_ratio} '
                  f'--save_fig True '
                  f'--save_path vis_{model_name}_{dataset_name}_4x_testset '
                  f'--cal_metrics True '
                  f'--dataset_name {dataset_name}'
                  )

        # os.system(f'CUDA_VISIBLE_DEVICES=5 python test_inr_liif_metasr_aliif.py '
        #           f'--config tools/paper_tools/vis_fixed_scale_UC_INR_liif_metasr_aliif.yaml '
        #           f'--model checkpoints/{exp}/{cp} '
        #           f'--scale_ratio {scale_ratio} '
        #           f'--save_fig True '
        #           f'--save_path vis_{model_name}_{dataset_name}_4x_testset '
        #           f'--cal_metrics True '
        #           f'--dataset_name {dataset_name}'
        #           )

os.system(f'zip -q -r vis_{model_name}_{dataset_name}_4x_testset.zip vis_{model_name}_{dataset_name}_4x_testset')
os.system(f'aws s3 cp vis_{model_name}_{dataset_name}_4x_testset.zip s3://xhs.bravo/user/kyanchen/tmp/')
