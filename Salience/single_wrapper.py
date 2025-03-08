import os
import logging
import math
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetFromList, MapDataset
from torch.utils.data import DataLoader


from detectron2.projects.deeplab import add_deeplab_config
import sys
sys.path.append("path/salience_llava/QAGNet_main")
from mask2former import add_maskformer2_config


from mask2former.data.dataset_mappers.assr_dataset_mapper import AssrDatasetMapper
from mask2former.data.dataset_mappers.irsr_dataset_mapper import IrsrDatasetMapper
from mask2former.data.dataset_mappers.sifr_dataset_mapper import SIFRdataDatasetMapper
from mask2former.data.datasets.register_sifr import get_sifr_dicts
from mask2former.data.datasets.register_assr import get_assr_dicts
from mask2former.data.datasets.register_irsr import get_irsr_dicts


from pycocotools import mask as coco_mask
from detectron2.structures import Instances


logger = logging.getLogger("detectron2")

def trivial_batch_collator(batch):
    """
    A trivial batch collator that simply returns
    the list of samples as is (no merging).
    """
    return batch

def convert_coco_poly_to_mask(segmentations, height, width):
    """
    Convert polygon segmentations to binary masks (uint8).
    """
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask_np = coco_mask.decode(rles)
        if len(mask_np.shape) < 3:
            mask_np = mask_np[..., None]
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.uint8)
        mask_tensor = mask_tensor.any(dim=2)
        masks.append(mask_tensor)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class SalienceSingleImageWrapper:
    """
    Runs a Detectron2-based Mask2Former (QAGNet) model on a dataset,
    one image at a time, returning a 'saliency map' for each call.
    """

    def __init__(self, config_file, resume=False, opts=None):
        """
        1) Build Detectron2 config
        2) Build the QAGNet/Mask2Former model
        3) Build the dataset & single-image dataloader
        4) Store an iterator to fetch images one by one
        """
        if opts is None:
            opts = []

        # Step 1: Build config
        self.cfg = self._setup_cfg(config_file, opts)

        # Step 2: Build model
        self.model = build_model(self.cfg)
        logger.info(f"Model:\n{self.model}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")

        # Load weights
        model_root_dir = self.cfg.EVALUATION.MODEL_DIR
        model_names = self.cfg.EVALUATION.MODEL_NAMES
        # If you have multiple model paths, pick the first for demonstration
        self.model_name = model_names[0]
        model_path = os.path.join(model_root_dir, self.model_name)
        DetectionCheckpointer(self.model, save_dir=model_root_dir).resume_or_load(
            model_path, resume=resume
        )

        # Step 3: Build dataset & single-batch dataloader
        self.datasetmode = self.cfg.EVALUATION.DATASETMODE
        self.dataloader = self._build_dataloader(self.cfg)

        # Step 4: Iterator to cycle through images
        self.data_iterator = iter(self.dataloader)

        # Misc settings from config
        self.result_threshold = self.cfg.EVALUATION.RESULT_THRESHOLD
        self.limited = self.cfg.EVALUATION.LIMITED
        self.dataset = self.cfg.EVALUATION.DATASET

        # Optional cap on how many images to process
        self.max_images = getattr(self.cfg.EVALUATION, "MAX_IMAGES", 200)
        self.processed_count = 0


    def _setup_cfg(self, config_file, opts):
        """
        Build and return the Detectron2 config.
        """
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)

        # Freeze and set up loggers
        cfg.freeze()
        default_setup(cfg, None)
        setup_logger(
            output=cfg.OUTPUT_DIR,
            distributed_rank=comm.get_rank(),
            name="mask2former"
        )
        return cfg


    def _build_dataloader(self, cfg):
        """
        Build the dataset (assr/sifr/irsr) and
        return a DataLoader with batch_size=1.
        """
        dataPath = cfg.EVALUATION.DATAPATH
        datasetmode = cfg.EVALUATION.DATASETMODE
        dataset_name = cfg.EVALUATION.DATASET

        if dataset_name == "assr":
            sor_dataset_path = os.path.join(dataPath, "ASSR/")
            print("Creating ASSR dataset ...")
            dataset_dicts = get_assr_dicts(root=sor_dataset_path, mode=datasetmode)
            mapper = AssrDatasetMapper(cfg, False)

        elif dataset_name == "sifr":
            sor_dataset_path = os.path.join(dataPath)
            print("Creating SIFR dataset ...")
            dataset_dicts = get_sifr_dicts(root=sor_dataset_path, mode=datasetmode)
            mapper = SIFRdataDatasetMapper(cfg, False)

        elif dataset_name == "irsr":
            sor_dataset_path = os.path.join(dataPath, "IRSR/")
            print("Creating IRSR dataset ...")
            dataset_dicts = get_irsr_dicts(root=sor_dataset_path, mode=datasetmode)
            mapper = IrsrDatasetMapper(cfg, False)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Turn the dataset dicts into a mapped dataset
        dataset_list = DatasetFromList(dataset_dicts, copy=False)
        mapped_dataset = MapDataset(dataset_list, mapper)

        # Single-image batches
        dataloader = DataLoader(
            mapped_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=trivial_batch_collator
        )
        return dataloader


    def get_next_saliency(self):
        """
        Grab the next sample from the dataset, run inference to
        produce a saliency map, and return a tensor [H, W].
        """
        if self.processed_count >= self.max_images:
            print("Reached the maximum number of images.")
            return None

        try:
            inputs = next(self.data_iterator)
        except StopIteration:
            print("No more data available in the dataset.")
            return None

        sal_map = self._inference_single(inputs)
        self.processed_count += 1
        return sal_map


    def _inference_single(self, inputs):
        """
        Perform single-image inference and return a saliency map
        as a torch.IntTensor [H, W].
        """
        input_ = inputs[0]  # batch_size=1, so it's a 1-element list
        img_height = input_["height"]
        img_width = input_["width"]

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(inputs)

        # If the model returns a list of dicts, take the last
        if isinstance(predictions, list):
            final_pred = predictions[-1]
        else:
            final_pred = predictions

        # If no instances => blank
        if "instances" not in final_pred:
            return torch.zeros((img_height, img_width), dtype=torch.int)

        instances = final_pred["instances"].to("cpu")

        # Confidence threshold
        thresholded = []
        for inst_idx in range(len(instances)):
            score = instances[inst_idx].scores
            if score > self.result_threshold:
                thresholded.append(instances[inst_idx])

        if len(thresholded) == 0:
            return torch.zeros((img_height, img_width), dtype=torch.int)

        # Combine all passing instances
        pred_instances = Instances.cat(thresholded)

        # Convert ground-truth polygons if needed (not strictly required)
        gt_polygons = [ann['segmentation'] for ann in input_['annotations']]
        gt_masks_tensor = convert_coco_poly_to_mask(gt_polygons, img_height, img_width)

        # Usually these come from your custom model (QAGNet):
        pred_masks_tensor = pred_instances.pred_masks.detach().cpu()
        # "pred_rank" is a custom attribute for ranking; may not exist in base Mask2Former
        pred_ranks_tensor = pred_instances.pred_rank.detach().cpu()

        # If user wants to limit # of instances
        if self.limited:
            if self.dataset == 'assr':
                top_k = 5
            elif self.dataset == 'irsr':
                top_k = 8
            elif self.dataset == 'sifr':
                top_k = 41
            else:
                top_k = pred_ranks_tensor.shape[0]

            if pred_ranks_tensor.shape[0] > top_k:
                values, indices = torch.topk(pred_ranks_tensor, top_k)
                mask_ = torch.zeros_like(pred_ranks_tensor, dtype=torch.bool)
                mask_[indices] = True
                pred_masks_tensor = pred_masks_tensor[mask_]
                pred_ranks_tensor = pred_ranks_tensor[mask_]

        # Start a blank map
        if gt_masks_tensor.shape[0] > 0:
            all_segmaps = torch.zeros_like(gt_masks_tensor[0], dtype=torch.float32)
        else:
            all_segmaps = torch.zeros((img_height, img_width), dtype=torch.float32)

        # Build rank-based coloring
        pred_ranks_list = pred_ranks_tensor.tolist()
        # For each rank, figure out relative ordering
        color_idx_list = [sorted(pred_ranks_list).index(a) + 1 for a in pred_ranks_list]

        color_values = torch.zeros(len(color_idx_list), dtype=torch.float32)
        if self.dataset == 'assr':
            # If <=10 objects, spread them 0..255 in steps
            if len(color_idx_list) <= 10:
                for i in range(len(color_idx_list)):
                    val = math.floor(
                        255.0 / 10 * (color_idx_list[i] + (10 - len(color_idx_list)))
                    )
                    color_values[i] = val
            else:
                # If more than 10, clamp the minimum color
                for i in range(len(color_idx_list)):
                    val = math.floor(
                        255.0 / 10 * (color_idx_list[i] + (10 - len(color_idx_list)))
                    )
                    color_values[i] = max(val, 25)
        else:
            # Generic spread for other datasets
            for i in range(len(color_idx_list)):
                val = (255.0 / len(color_idx_list)) * color_idx_list[i]
                color_values[i] = val

        cover_region = (all_segmaps != 0)
        color_idx_tensor = torch.tensor(color_idx_list, dtype=torch.int32)
        segmaps_pred = pred_masks_tensor

        # We iterate in descending rank to "overlay" correctly
        max_idx = segmaps_pred.shape[0]
        for step_val in range(max_idx, 0, -1):
            obj_id_list_tensor = (color_idx_tensor == step_val).nonzero().flatten()
            if obj_id_list_tensor.numel() == 0:
                continue
            for obj_id in obj_id_list_tensor:
                seg = segmaps_pred[obj_id].float()
                cval = color_values[obj_id]
                mask_ = seg >= 0.5
                seg[mask_] = cval
                seg[~mask_] = 0.0
                # zero out overlap
                seg[cover_region] = 0.0
                all_segmaps += seg
                cover_region = (all_segmaps != 0)

        # Return an IntTensor saliency
        return all_segmaps.to(torch.int)


if __name__ == "__main__":
    """
    Minimal usage example if you run this file directly:
    1) Build the wrapper
    2) Repeatedly fetch saliency for each image in the dataset
    """
    config_file = "path to the config file"

    wrapper = SalienceSingleImageWrapper(config_file=config_file, resume=False, opts=[])
    while True:
        sal_map = wrapper.get_next_saliency()
        if sal_map is None:
            print("No more saliency maps or limit reached.")
            break
        print("Saliency map shape:", sal_map.shape)
