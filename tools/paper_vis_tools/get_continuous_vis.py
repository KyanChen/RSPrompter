import os

# os.system('cd ..')
exp = 'EXP20221219_1'

for cp in ['epoch-last.pth']:
    for scale_ratio in [1.5, 2, 2.5, 3, 3.5, 4.0]:
        print(cp, '   ', scale_ratio)

        os.system(f'CUDA_VISIBLE_DEVICES=1 python test_inr_diinn_arbrcan_sadnarc_funsr_overnet.py '
              f'--config tools/paper_tools/vis_continuous_UC_INR_diinn_arbrcan_funsr_overnet.yaml '
              f'--model checkpoints/{exp}/{cp} '
              f'--scale_ratio {scale_ratio} ' 
              f'--save_fig True '
              f'--save_path vis_AID_testset '
              f'--cal_metrics False'
                  )

        print('*' * 30)
