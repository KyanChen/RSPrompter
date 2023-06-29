from typing import List

from mmpl.registry import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class ISAIDInsSegDataset(CocoDataset):
    """Dataset for Cityscapes."""
    # Large_Vehicle Small_Vehicle ship
    # METAINFO = {
    #     'classes': ['ship', 'storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter', 'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor'],
    #     'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    #                 (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
    #                 (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
    #                 (0, 127, 191), (  0,127,255), (  0,100,155)
    #                 ]
    # }

    METAINFO = {
        'classes': ['storage_tank', 'baseball_diamond', 'tennis_court', 'basketball_court',
                    'Ground_Track_Field', 'Bridge', 'Helicopter', 'Swimming_pool',
                    'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                    (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
                    (0, 127, 191), (0, 127, 255), (0, 100, 155)
                    ]
    }

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        # if self.test_mode:
        #     return self.data_list

        # if self.filter_cfg is None:
        #     return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            all_is_crowd = all([
                instance['ignore_flag'] == 1
                for instance in data_info['instances']
            ])
            if filter_empty_gt and (img_id not in ids_in_cat or all_is_crowd):
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
