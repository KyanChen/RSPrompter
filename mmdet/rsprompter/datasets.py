from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class NWPUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['airplane', 'ship', 'storage_tank', 'baseball_diamond',
                    'tennis_court', 'basketball_court', 'ground_track_field',
                    'harbor', 'bridge', 'vehicle'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230),
                    (119, 11, 32), (0, 255, 0), (0, 0, 255)]
    }


@DATASETS.register_module()
class WHUInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['building'],
        'palette': [(0, 255, 0)]
    }


@DATASETS.register_module()
class SSDDInsSegDataset(CocoDataset):
    METAINFO = {
        'classes': ['ship'],
        'palette': [(0, 0, 255)]
    }