import os

# os.system('cd ..')
exp = 'EXP20221219_3'

# for cp in ['epoch-best.pth', 'epoch-last.pth'] + [f'epoch-{x}.pth' for x in range(300, 4000, 300)]:
for cp in ['epoch-best.pth', 'epoch-last.pth']:
# for cp in ['epoch-last.pth']:
# for cp in ['epoch-best.pth']:
# for cp in ['epoch-300.pth']:
    for scale_ratio in [2, 2.5, 3, 3.5, 4.0, 6, 8, 10]:
    # for scale_ratio in [4]:
        print(cp, '   ', scale_ratio)

        # os.system(f'CUDA_DEVICES_VISIBLE=0 python test_interpolate_sr.py '
        #           f'--config configs/test_interpolate.yaml '
        #           f'--scale_ratio {scale_ratio}')

        # os.system(f'CUDA_VISIBLE_DEVICES=1 python test_inr_liif_metasr_aliif.py '
        #           f'--config configs/baselines/test_INR_liif_metasr_aliif.yaml '
        #           f'--model checkpoints/{exp}/{cp} '
        #           f'--scale_ratio {scale_ratio}')

        # os.system(f'CUDA_VISIBLE_DEVICES=7 python test_inr_mysr.py '
        #           f'--config configs/test_UC_INR_mysr.yaml '
        #           f'--model checkpoints/{exp}/{cp} '
        #           f'--scale_ratio {scale_ratio}')

        os.system(f'CUDA_VISIBLE_DEVICES=1 python test_inr_diinn_arbrcan_sadnarc_funsr_overnet.py '
              f'--config configs/test_INR_diinn_arbrcan_funsr_overnet.yaml '
              f'--model checkpoints/{exp}/{cp} '
              f'--scale_ratio {scale_ratio} '
              f'--dataset_name AID')

        # os.system(f'CUDA_DEVICES_VISIBLE=0 python test_cnn_sr.py '
        #       f'--config configs/test_CNN.yaml '
        #       f'--model checkpoints/{exp}/{cp} '
        #       f'--scale_ratio {scale_ratio}')

        print('*' * 30)
