

import sys
import os
import logging

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger

# Project-specific imports
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

# Import your salience inference function
from inference_salience import inference

import pdb
logger = logging.getLogger("detectron2")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

def main(args):
    """
    Original main function for command-line usage.
    You can still run this file directly, e.g.:
        python retrieve_salience.py --num-gpus 1 ...
    """
    cfg = setup(args)
    model = build_model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    logger.info("Model:\n{}".format(model))

    # Retrieve config settings for inference
    model_root_dir = cfg.EVALUATION.MODEL_DIR
    datasetmode = cfg.EVALUATION.DATASETMODE
    model_names = cfg.EVALUATION.MODEL_NAMES

    if comm.is_main_process():
        for model_name in model_names:
            model_path = os.path.join(model_root_dir, model_name)
            DetectionCheckpointer(model, save_dir=model_root_dir).resume_or_load(
                model_path, resume=args.resume
            )

            # Run inference, which returns a list of saliency maps
            saliency_maps = inference(cfg, model, model_name, model_root_dir, datasetmode)

            # Print shape of each saliency map
            for idx, s_map in enumerate(saliency_maps):
                print(f"[{model_name}] Saliency map {idx}: shape = {s_map.shape}")

def wrapper_mask():
    """
    A simple wrapper function that:
      1) Builds a default 'args' object (no command-line parsing).
      2) Builds the config & model.
      3) Loads checkpoints & calls inference for *each* MODEL_NAME.
      4) RETURNS a single list of saliency map tensors.
         (All saliency maps from all model_names are appended into one list.)
    
    Usage example in another script:
        from retrieve_salience import wrapper_mask
        saliency_list = wrapper_mask()
        # saliency_list is [torch.Tensor, torch.Tensor, ...]
        # each element is shape (H, W).
    """
    # 1) Create a minimal "args" object with defaults
    class DummyArgs:
        config_file = "./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
        resume = False
        opts = []
        num_gpus = 1
        machine_rank = 0
        dist_url = "auto"

    args = DummyArgs()

    # 2) Build config & model
    cfg = setup(args)
    model = build_model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    logger.info("Model:\n{}".format(model))

    # 3) Retrieve config settings
    model_root_dir = cfg.EVALUATION.MODEL_DIR
    datasetmode = cfg.EVALUATION.DATASETMODE
    model_names = cfg.EVALUATION.MODEL_NAMES

    # We'll collect *all* saliency maps from *all* models in a single list:
    combined_saliency = []

    # 4) Loop through each model and run inference
    if comm.is_main_process():
        for model_name in model_names:
            model_path = os.path.join(model_root_dir, model_name)
            DetectionCheckpointer(model, save_dir=model_root_dir).resume_or_load(
                model_path, resume=args.resume
            )
            pdb.set_trace()

            saliency_maps = inference(cfg, model, model_name, model_root_dir, datasetmode)

            # Append these maps to the combined list
            combined_saliency.extend(saliency_maps)

            # (Optional) Print info
            for idx, s_map in enumerate(saliency_maps):
                print(f"[{model_name}] Saliency map {idx}: shape = {s_map.shape}")

    # 5) Return the single combined list of saliency map tensors
    return combined_saliency


if __name__ == "__main__":
    # Standard detectron2 argument parser for CLI usage
    parser = default_argument_parser()
    args = parser.parse_args()

    # Override or set additional defaults as needed:
    args.config_file = "./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
    args.resume = False

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
