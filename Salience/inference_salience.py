import time
import os
from contextlib import contextmanager
import torch
from tqdm import tqdm
import copy
import cv2

# External modules (you must have these in your environment)
from rankevaluation.SASOR import evalu as rank_evalu
from rankevaluation.mae_fmeasure_2 import evalu as mf_evalu
import pickle as pkl
from detectron2.data import build_detection_test_loader
from mask2former.data.dataset_mappers.assr_dataset_mapper import AssrDatasetMapper
from mask2former.data.dataset_mappers.irsr_dataset_mapper import IrsrDatasetMapper
from mask2former.data.dataset_mappers.sifr_dataset_mapper import SIFRdataDatasetMapper
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from pycocotools import mask as coco_mask
from rankevaluation.sor_eval import doOffcialSorInference
import detectron2.utils.comm as comm

from mask2former.data.datasets.register_sifr import get_sifr_dicts
from mask2former.data.datasets.register_assr import get_assr_dicts
from mask2former.data.datasets.register_irsr import get_irsr_dicts
from detectron2.data import DatasetFromList, MapDataset

from torch.utils.data import DataLoader, SequentialSampler
import math
from thop import profile

import pdb
import matplotlib as plt


def find_all_indexes(lst, value):
    """
    Return a list of all indexes in 'lst' that match 'value'.
    (Used for color assignment logic in the original code.)
    """
    return [i for i, v in enumerate(lst) if v == value]


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    Convert polygon segmentations from COCO format into binary masks (uint8 tensors).
    segmentations: list of polygons, each polygon is [ [x0, y0, x1, y1, ...], ... ]
    height, width: image dimensions
    """
    masks = []
    for polygons in segmentations:
        # polygons is expected to be a list of lists of coordinates
        rles = coco_mask.frPyObjects(polygons, height, width)
        # decode() returns a NumPy array, so convert to torch afterward
        mask_np = coco_mask.decode(rles)  # shape: (H, W) or (H, W, #polygons)
        if len(mask_np.shape) < 3:
            # shape = (H, W) => single object
            mask_np = mask_np[..., None]
        # Convert to torch tensor
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.uint8)
        # Combine polygons along dim=2
        mask_tensor = mask_tensor.any(dim=2)
        masks.append(mask_tensor)

    if masks:
        masks = torch.stack(masks, dim=0)  # shape: (N, H, W)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


@contextmanager
def inference_context(model):
    """
    Temporarily set model to eval mode, then restore previous mode afterwards.
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def trivial_batch_collator(batch):
    """
    A trivial batch collator that simply returns the list of samples.
    """
    return batch


def inference(cfg, model, model_name, model_root_dir, datasetmode):
    """
    Generate saliency maps for each image in the dataset, return them as a list of PyTorch tensors.
    Each tensor has shape (H, W), dtype=int. One tensor per image.

    Args:
        cfg:            detectron2 config
        model:          trained model
        model_name:     name/identifier of the model file
        model_root_dir: path to the directory containing the model
        datasetmode:    "test", "val", etc. for dataset split

    Returns:
        all_saliency_maps (list of torch.Tensor): one [H, W] tensor per image
    """
    if not comm.is_main_process():
        # If not the main process, do nothing
        return []

    dataset = cfg.EVALUATION.DATASET
    limited = cfg.EVALUATION.LIMITED
    dataPath = cfg.EVALUATION.DATAPATH

    # ---- Build the dataset & DataLoader ----
    if dataset == "assr":
        SOR_DATASETPATH = os.path.join(dataPath, "ASSR/")
        print('------Evaluation based on ASSR dataset!------')
        assr_dataset = get_assr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
        assr_dataset_list = DatasetFromList(assr_dataset, copy=False)
        assr_dataset_list = MapDataset(assr_dataset_list, AssrDatasetMapper(cfg, False))
        dataloader = DataLoader(
            assr_dataset_list, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=trivial_batch_collator
        )
    elif dataset == 'sifr':
        SOR_DATASETPATH = os.path.join(dataPath)
        print('------Evaluation based on SIFR dataset!------')
        sifr_dataset = get_sifr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
        sifr_dataset_list = DatasetFromList(sifr_dataset, copy=False)
        sifr_dataset_list = MapDataset(sifr_dataset_list, SIFRdataDatasetMapper(cfg, False))
        dataloader = DataLoader(
            sifr_dataset_list, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=trivial_batch_collator
        )
        pdb.set_trace()
    elif dataset == 'irsr':
        SOR_DATASETPATH = os.path.join(dataPath, "IRSR/")
        print('------Evaluation based on IRSR dataset!------')
        irsr_dataset = get_irsr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
        irsr_dataset_list = DatasetFromList(irsr_dataset, copy=False)
        irsr_dataset_list = MapDataset(irsr_dataset_list, IrsrDatasetMapper(cfg, False))
        dataloader = DataLoader(
            irsr_dataset_list, batch_size=1, shuffle=False,
            num_workers=0, collate_fn=trivial_batch_collator
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Where to save the PNG images (saliency maps)
    output_dir = os.path.join('./evaluationResult', model_name.split('.')[0])
    saliency_map_dir = os.path.join(
        output_dir,
        'GeneratedSaliencyMaps',
        f"{model_name}-ResultThres-{cfg.EVALUATION.RESULT_THRESHOLD}Limited-{limited}"
    )
    os.makedirs(saliency_map_dir, exist_ok=True)
    print("------Saliency map path is:", saliency_map_dir)

    # We'll store each final saliency map as a Torch tensor (H, W)
    all_saliency_maps = []

    # Control how many images we process
    # MAX_IMAGES = cfg.EVALUATION.MAX_IMAGES if hasattr(cfg.EVALUATION, "MAX_IMAGES") else 16
    # print(f"Will process up to {MAX_IMAGES} images...")

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            name = os.path.basename(inputs[0]["file_name"])
            print(name)
            # if idx >= MAX_IMAGES:
            #     print(f"Reached {MAX_IMAGES} images; stopping.")
            #     break
            predictions = model(inputs)
            img_height = inputs[0]["height"]
            img_width = inputs[0]["width"]
            name = os.path.basename(inputs[0]["file_name"])
            # Get final predictions
            if isinstance(predictions, list):
                final_pred = predictions[-1]
            else:
                final_pred = predictions
            # If no "instances" in final_pred, produce a blank map
            if "instances" not in final_pred:
                all_segmaps = torch.zeros((img_height, img_width), dtype=torch.float32)
                all_segmaps_int = all_segmaps.to(torch.int)
                cv2.imwrite(
                    os.path.join(saliency_map_dir, f'{os.path.splitext(name)[0]}.png'),
                    all_segmaps_int.cpu().numpy()
                )
                all_saliency_maps.append(all_segmaps_int)
                continue

            # Retrieve predicted instances
            instances = final_pred["instances"].to("cpu")

            # Threshold by confidence
            thresholded_instances = []
            for instance_idx in range(len(instances)):
                score = instances[instance_idx].scores
                if score > cfg.EVALUATION.RESULT_THRESHOLD:
                    thresholded_instances.append(instances[instance_idx])

            # If no predicted masks remain after threshold => blank map
            if len(thresholded_instances) == 0:
                all_segmaps = torch.zeros((img_height, img_width), dtype=torch.float32)
                all_segmaps_int = all_segmaps.to(torch.int)
                cv2.imwrite(
                    os.path.join(saliency_map_dir, f'{os.path.splitext(name)[0]}.png'),
                    all_segmaps_int.cpu().numpy()
                )
                all_saliency_maps.append(all_segmaps_int)
                continue

            pred_instances = Instances.cat(thresholded_instances)

            # Convert GT polygons to mask just in case you need them
            gt_masks_polygon = []
            for ann in inputs[0]['annotations']:
                gt_masks_polygon.append(ann['segmentation'])
            gt_masks_tensor = convert_coco_poly_to_mask(gt_masks_polygon, img_height, img_width)

            # Extract predicted masks & ranks
            pred_masks_tensor = pred_instances.pred_masks.detach().cpu()
            pred_ranks_tensor = pred_instances.pred_rank.detach().cpu()

            # If "limited" is True, apply top-K logic
            if limited:
                if dataset == 'assr':
                    top_k = 5
                elif dataset == 'irsr':
                    top_k = 8
                elif dataset == 'sifr':
                    top_k = 41
                else:
                    top_k = pred_ranks_tensor.shape[0]
                
                if pred_ranks_tensor.shape[0] > top_k:
                    values, indices = torch.topk(pred_ranks_tensor, top_k)
                    mask = torch.zeros_like(pred_ranks_tensor, dtype=torch.bool)
                    mask[indices] = True
                    pred_masks_tensor = pred_masks_tensor[mask]
                    pred_ranks_tensor = pred_ranks_tensor[mask]

            # Create a blank saliency map
            if gt_masks_tensor.shape[0] > 0:
                all_segmaps = torch.zeros_like(gt_masks_tensor[0], dtype=torch.float32)
            else:
                all_segmaps = torch.zeros((img_height, img_width), dtype=torch.float32)

            # Build rank-based coloring
            pred_ranks_list = pred_ranks_tensor.tolist()
            sorted_ranks, sorted_idx = torch.sort(pred_ranks_tensor, dim=0)
            color_idx_list = [sorted(pred_ranks_list).index(a) + 1 for a in pred_ranks_list]

            # Compute color values
            color_values = torch.zeros(len(color_idx_list), dtype=torch.float32)
            if dataset == 'assr':
                # mapping for <=10 objects
                if len(color_idx_list) <= 10:
                    for i in range(len(color_idx_list)):
                        val = math.floor(255.0 / 10 * (color_idx_list[i] + (10 - len(color_idx_list))))
                        color_values[i] = val
                else:
                    for i in range(len(color_idx_list)):
                        val = math.floor(255.0 / 10 * (color_idx_list[i] + (10 - len(color_idx_list))))
                        color_values[i] = max(val, 25)
            else:
                # simple spread from 0 to 255
                for i in range(len(color_idx_list)):
                    val = (255.0 / len(color_idx_list)) * color_idx_list[i]
                    color_values[i] = val

            segmaps_pred = pred_masks_tensor
            if segmaps_pred.shape[0] != 0:
                cover_region = (all_segmaps != 0)
                color_idx_tensor = torch.tensor(color_idx_list, dtype=torch.int32)

                max_idx = segmaps_pred.shape[0]
                # We iterate in reverse order of rank
                for step_val in range(max_idx, 0, -1):
                    obj_id_list_tensor = (color_idx_tensor == step_val).nonzero().flatten()
                    if obj_id_list_tensor.numel() == 0:
                        continue
                    for obj_id in obj_id_list_tensor:
                        seg = segmaps_pred[obj_id].clone().float()
                        cval = color_values[obj_id]
                        mask_ = seg >= 0.5
                        seg[mask_] = cval
                        seg[~mask_] = 0.0
                        # zero out where covered
                        seg[cover_region] = 0.0
                        all_segmaps += seg
                        cover_region = (all_segmaps != 0)

            # Convert to int, save to disk, and keep it
            all_segmaps_int = all_segmaps.to(torch.int)
            all_segmaps_np = all_segmaps_int.cpu().numpy()
            cv2.imwrite(
                os.path.join(saliency_map_dir, f'{os.path.splitext(name)[0]}.png'),
                all_segmaps_np
            )

            all_saliency_maps.append(all_segmaps_int)

    # Return a list of PyTorch tensors, each is shape [H, W], dtype=int
    return all_saliency_maps