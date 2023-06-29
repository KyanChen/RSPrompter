from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
import torch.distributed as dist
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.utilities.distributed import gather_all_tensors
from mmpl.registry import METRICS


@METRICS.register_module(force=True)
class PLMeanAveragePrecision(MeanAveragePrecision):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def _sync_dist(self, dist_sync_fn: Callable = gather_all_tensors, process_group: Optional[Any] = None) -> None:
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)

        if self.iou_type == "segm":
            self.detections = self._gather_tuple_list(self.detections, process_group)
            self.groundtruths = self._gather_tuple_list(self.groundtruths, process_group)

    @staticmethod
    def _gather_tuple_list(list_to_gather: List[Tuple], process_group: Optional[Any] = None) -> List[Any]:
        world_size = dist.get_world_size(group=process_group)
        list_gathered = [None] * world_size
        dist.all_gather_object(list_gathered, list_to_gather, group=process_group)

        for rank in range(1, world_size):
            assert (
                len(list_gathered[rank]) == list_gathered[0],
                f"Rank{rank} doesn't have the same number of elements as Rank0: "
                f"{list_gathered[rank]} vs. {list_gathered[0]}",
            )
        list_merged = []
        for idx in range(len(list_gathered[0])):
            for rank in range(world_size):
                list_merged.append(list_gathered[rank][idx])

        return list_merged
