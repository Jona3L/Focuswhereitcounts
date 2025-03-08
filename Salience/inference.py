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
    Run inference on a given dataset, produce saliency maps, and optionally compute metrics.
    We'll only process the first N images (set to 1 for demonstration).

    Args:
        cfg:            detectron2 config
        model:          your trained model
        model_name:     name/identifier of the model file
        model_root_dir: path to the directory containing the model (not used in this snippet)
        datasetmode:    typically "test", "val", etc., used to select the dataset split
    """
    # Only run on main process (for distributed training setups)
    if not comm.is_main_process():
        return

    dataset = cfg.EVALUATION.DATASET
    limited = cfg.EVALUATION.LIMITED
    dataPath = cfg.EVALUATION.DATAPATH

    # 1) Build the dataset & DataLoader
    if dataset == "assr":
        SOR_DATASETPATH = os.path.join(dataPath, "ASSR/")
        print('------Evaluation based on ASSR dataset!------')

        assr_dataset = get_assr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
        assr_dataset_list = DatasetFromList(assr_dataset, copy=False)
        assr_dataset_list = MapDataset(assr_dataset_list, AssrDatasetMapper(cfg, False))
        dataloader = DataLoader(
            assr_dataset_list,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=trivial_batch_collator
        )

    elif dataset == 'sifr':
        SOR_DATASETPATH = os.path.join(dataPath, "SIFR/")
        print('------Evaluation based on SIFR dataset!------')

        sifr_dataset = get_sifr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
        sifr_dataset_list = DatasetFromList(sifr_dataset, copy=False)
        sifr_dataset_list = MapDataset(sifr_dataset_list, SIFRdataDatasetMapper(cfg, False))
        dataloader = DataLoader(
            sifr_dataset_list,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=trivial_batch_collator
        )

    elif dataset == 'irsr':
        SOR_DATASETPATH = os.path.join(dataPath, "IRSR/")
        print('------Evaluation based on IRSR dataset!------')

        irsr_dataset = get_irsr_dicts(root=SOR_DATASETPATH, mode=datasetmode)
        irsr_dataset_list = DatasetFromList(irsr_dataset, copy=False)
        irsr_dataset_list = MapDataset(irsr_dataset_list, IrsrDatasetMapper(cfg, False))
        dataloader = DataLoader(
            irsr_dataset_list,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=trivial_batch_collator
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # 2) Prepare output folder for saliency maps
    output_dir = os.path.join('./evaluationResult', model_name.split('.')[0])
    saliency_map_dir = os.path.join(
        output_dir,
        'GeneratedSaliencyMaps',
        f"{model_name}-ResultThres-{cfg.EVALUATION.RESULT_THRESHOLD}Limited-{limited}"
    )
    print("------Saliency map path is :", saliency_map_dir)
    os.makedirs(saliency_map_dir, exist_ok=True)

    # We'll only process the first 1 image for demonstration
    MAX_IMAGES = 1

    # This will store results for potential metric calculations
    res = []

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            if idx >= MAX_IMAGES:
                print(f"Reached {MAX_IMAGES} images; stopping early.")
                break

            # 3) Forward pass
            predictions = model(inputs)  # prediction is a list of dict or single dict
            img_height = inputs[0]["height"]
            img_width = inputs[0]["width"]

            # Some models return a list of stages, some return single dict
            # Here we assume the final stage is predictions[-1].
            if isinstance(predictions, list):
                final_pred = predictions[-1]
            else:
                final_pred = predictions

            if "instances" in final_pred:
                instances = final_pred["instances"].to("cpu")
                pred_instances = Instances(instances.image_size)

                # 4) Filter out predictions below threshold
                thresholded_instances = []
                for instance_idx in range(len(instances)):
                    score = instances[instance_idx].scores
                    if score > cfg.EVALUATION.RESULT_THRESHOLD:
                        thresholded_instances.append(instances[instance_idx])

                # If after thresholding we have predictions
                if len(thresholded_instances) > 0:
                    pred_instances = Instances.cat(thresholded_instances)

                # 5) Load ground-truth polygons, convert to mask tensors
                gt_masks_polygon = []
                gt_ranks = []
                for ann in inputs[0]['annotations']:
                    gt_masks_polygon.append(ann['segmentation'])
                    gt_ranks.append(ann['gt_rank'])

                gt_masks_tensor = convert_coco_poly_to_mask(gt_masks_polygon, img_height, img_width)

                # Extract image file name
                name = os.path.basename(inputs[0]["file_name"])

                # Check if we have predictions above threshold
                if len(thresholded_instances) > 0:
                    # 6) Keep masks & ranks in Tensors
                    pred_masks_tensor = pred_instances.pred_masks.detach().cpu()  # (N_pred, H, W)
                    pred_ranks_tensor = pred_instances.pred_rank.detach().cpu()   # (N_pred,)
                    pdb.set_trace()
                    pred_ranks_list = pred_ranks_tensor.tolist()                  # python list

                    # 7) (Optional) limit top-K using all-Tensor ops
                    if limited:
                        if dataset == 'assr':
                            # top-5
                            top_k = 5
                        elif dataset == 'irsr':
                            # top-8
                            top_k = 8
                        elif dataset == 'sifr':
                            # top-41
                            top_k = 41
                        else:
                            top_k = pred_ranks_tensor.shape[0]  # no limit

                        if pred_ranks_tensor.shape[0] > top_k:
                            values, indices = torch.topk(pred_ranks_tensor, top_k)
                            mask = torch.zeros_like(pred_ranks_tensor, dtype=torch.bool)
                            mask[indices] = True
                            pred_masks_tensor = pred_masks_tensor[mask]
                            pred_ranks_tensor = pred_ranks_tensor[mask]
                            pred_ranks_list = pred_ranks_tensor.tolist()

                    # 8) Prepare a blank saliency map in Tensor
                    if gt_masks_tensor.shape[0] > 0:
                        # Use the shape of the first GT mask
                        all_segmaps = torch.zeros_like(gt_masks_tensor[0], dtype=torch.float32)
                    else:
                        # If no GT, fallback to entire image dims
                        all_segmaps = torch.zeros((img_height, img_width), dtype=torch.float32)

                    # 9) Save results for potential metric calc
                    res.append({
                        'gt_masks': gt_masks_tensor,        # shape: (N_gt, H, W)
                        'segmaps': pred_masks_tensor,       # shape: (N_pred, H, W)
                        'gt_ranks': gt_ranks,               # list of GT ranks
                        'rank_scores': pred_ranks_tensor,   # shape: (N_pred,)
                        'img_name': name
                    })

                    # 10) Build the final saliency map
                    # Rank logic: "lowest rank -> index 1, next -> index 2, etc."
                    # color_idx_list: each predicted object’s 'ranking index'
                    #   we first find the sorted order in ascending rank
                    #   then each object's color_idx = position in the sorted list + 1
                    sorted_ranks, sorted_idx = torch.sort(pred_ranks_tensor, dim=0)  # ascending
                    # We can do it purely in Python:
                    color_idx_list = [sorted(pred_ranks_list).index(a) + 1 for a in pred_ranks_list]
                    color_len = len(color_idx_list)

                    # Compute color_values for each object (based on dataset rules)
                    color_values = torch.zeros(color_len, dtype=torch.float32)
                    if dataset == 'assr':
                        # If we have <=10 predicted objects, map them from 25->255 in steps
                        # else clamp the minimum color to 25
                        if color_len <= 10:
                            for i in range(color_len):
                                val = math.floor(255.0 / 10 * (color_idx_list[i] + (10 - color_len)))
                                color_values[i] = val
                        else:
                            for i in range(color_len):
                                val = math.floor(255.0 / 10 * (color_idx_list[i] + (10 - color_len)))
                                color_values[i] = max(val, 25)
                    else:
                        # For SIFR/IRSR or others, just spread from 0->255
                        for i in range(color_len):
                            val = (255.0 / color_len) * color_idx_list[i]
                            color_values[i] = val

                    segmaps_pred = pred_masks_tensor  # (N_pred, H, W)

                    if segmaps_pred.shape[0] != 0:
                        # We'll track covered region with a boolean mask
                        cover_region = (all_segmaps != 0)
                        color_idx_tensor = torch.tensor(color_idx_list, dtype=torch.int32)

                        # Reverse iteration to replicate "from highest color_idx down to 1"
                        max_idx = segmaps_pred.shape[0]
                        for step_val in range(max_idx, 0, -1):
                            # Find objects where color_idx == step_val
                            obj_id_list_tensor = (color_idx_tensor == step_val).nonzero().flatten()
                            if obj_id_list_tensor.numel() == 0:
                                continue

                            for obj_id in obj_id_list_tensor:
                                seg = segmaps_pred[obj_id].clone().float()  # (H, W)
                                cval = color_values[obj_id]                 # float
                                mask_ = seg >= 0.5
                                # Assign color cval to the mask pixels
                                seg[mask_] = cval
                                seg[~mask_] = 0.0

                                # zero out where covered_region is True
                                seg[cover_region] = 0.0

                                # accumulate into all_segmaps
                                all_segmaps += seg
                                cover_region = (all_segmaps != 0)

                        # Convert to int -> NumPy for cv2.imwrite
                        all_segmaps_int = all_segmaps.to(torch.int)
                        all_segmaps_np = all_segmaps_int.cpu().numpy()
                        pdb.set_trace()
                        cv2.imwrite(
                            os.path.join(saliency_map_dir, f'{os.path.splitext(name)[0]}.png'),
                            all_segmaps_np
                        )
                    else:
                        # If no predicted masks after threshold
                        all_segmaps_int = all_segmaps.to(torch.int)
                        all_segmaps_np = all_segmaps_int.cpu().numpy()
                        cv2.imwrite(
                            os.path.join(saliency_map_dir, f'{os.path.splitext(name)[0]}.png'),
                            all_segmaps_np
                        )

                else:
                    # No predictions above threshold => blank
                    name = os.path.basename(inputs[0]["file_name"])
                    if gt_masks_tensor.shape[0] > 0:
                        all_segmaps = torch.zeros_like(gt_masks_tensor[0], dtype=torch.int)
                    else:
                        all_segmaps = torch.zeros((img_height, img_width), dtype=torch.int)

                    all_segmaps_np = all_segmaps.cpu().numpy()
                    cv2.imwrite(
                        os.path.join(saliency_map_dir, f'{os.path.splitext(name)[0]}.png'),
                        all_segmaps_np
                    )

                    # Store empty segmaps for metrics
                    empty_segmap = torch.zeros((0, img_height, img_width), dtype=torch.uint8)
                    res.append({
                        'gt_masks': gt_masks_tensor,
                        'segmaps': empty_segmap,  # empty
                        'gt_ranks': gt_ranks,
                        'rank_scores': torch.tensor([], dtype=torch.float),
                        'img_name': name
                    })
            else:
                # If the model didn't produce "instances" at all, handle gracefully
                name = os.path.basename(inputs[0]["file_name"])
                # Convert GT anyway
                gt_masks_polygon = []
                gt_ranks = []
                for ann in inputs[0]['annotations']:
                    gt_masks_polygon.append(ann['segmentation'])
                    gt_ranks.append(ann['gt_rank'])

                gt_masks_tensor = convert_coco_poly_to_mask(gt_masks_polygon, img_height, img_width)

                if gt_masks_tensor.shape[0] > 0:
                    all_segmaps = torch.zeros_like(gt_masks_tensor[0], dtype=torch.int)
                else:
                    all_segmaps = torch.zeros((img_height, img_width), dtype=torch.int)

                all_segmaps_np = all_segmaps.cpu().numpy()
                cv2.imwrite(
                    os.path.join(saliency_map_dir, f'{os.path.splitext(name)[0]}.png'),
                    all_segmaps_np
                )

                empty_segmap = torch.zeros((0, img_height, img_width), dtype=torch.uint8)
                res.append({
                    'gt_masks': gt_masks_tensor,
                    'segmaps': empty_segmap,
                    'gt_ranks': gt_ranks,
                    'rank_scores': torch.tensor([], dtype=torch.float),
                    'img_name': name
                })

    # Optionally, compute metrics here with `res` or just return
    print("Finished generating saliency maps. Skipping metric calculations in this snippet.")
    return res
