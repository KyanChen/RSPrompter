import argparse
import glob
import os

import mmcv
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert iNaturalist2018 annotations to MMPretrain format.'
    )
    parser.add_argument('--input', type=str, default='/data/kyanchen/cache_data/ins_seg/sam_cls/whu/train/', help='input path')
    parser.add_argument('--output', type=str, default='/data/kyanchen/cache_data/ins_seg/sam_cls/whu/samcls_train.txt', help='Output list file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = glob.glob(args.input + '/*.jpg')
    output_lines = []
    for img_item in tqdm.tqdm(data):
        category_id = int(os.path.basename(img_item).split('_')[-1].split('.')[0])
        category_id = 1 if category_id == 255 else category_id
        file_name = os.path.basename(img_item)
        output_lines.append(f"{file_name} {category_id}\n")
    assert len(output_lines) == len(data)
    with open(args.output, 'w') as f:
        f.writelines(output_lines)


if __name__ == '__main__':
    main()
