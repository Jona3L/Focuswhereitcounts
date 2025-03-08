#!/usr/bin/env python3

import argparse
import torch
import requests
from io import BytesIO
from PIL import Image
import json
import os

import evaluate
from tqdm import tqdm  # For the progress bar and safe printing

import transformers
from transformers import TextStreamer
from peft import PeftModel

from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# Saliency model and wrapper
from llava.model.language_model.salience_llava import MySaliencyLlamaForCausalLM
from QAGNet_main.single_wrapper import SalienceSingleImageWrapper


###############################################################################
# 1. Utility Functions
###############################################################################

def load_image(image_file):
    """
    Load an image from a local path or a URL.
    """
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_annotations(annotations_json):
    """
    Loads a list of dict items from JSON, each item has:
        {
          "id": "...",
          "image": "...",
          "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt",   "value": "... ground truth caption ..."}
          ]
        }
    """
    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_ground_truth_caption(conversations):
    """
    Returns the last GPT response as the ground truth.
    Adjust if needed.
    """
    for entry in reversed(conversations):
        if entry["from"].lower() == "gpt":
            return entry["value"].strip()
    return ""


###############################################################################
# 2. Main Evaluation
###############################################################################

def main(args):
    # -------------------------------------------------------------------------
    # 2a. Prepare evaluation metrics
    # -------------------------------------------------------------------------
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    all_predictions = []
    all_references = []

    # -------------------------------------------------------------------------
    # 2b. Load the model
    # -------------------------------------------------------------------------
    disable_torch_init()

    load_kwargs = {}
    if args.load_4bit or args.load_8bit:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.float32
        if args.bf16:
            compute_dtype = torch.bfloat16
        elif args.fp16:
            compute_dtype = torch.float16

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_4bit,
            load_in_8bit=args.load_8bit,
            llm_int8_skip_modules=["mm_projector"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        load_kwargs["device_map"] = {"": args.device}
        load_kwargs["torch_dtype"] = compute_dtype
    else:
        # Possibly float16 or bfloat16
        if args.fp16:
            load_kwargs["torch_dtype"] = torch.float16
        elif args.bf16:
            load_kwargs["torch_dtype"] = torch.bfloat16

    print(f"[INFO] Loading MySaliencyLlamaForCausalLM from: {args.model_path}")
    model = MySaliencyLlamaForCausalLM.from_pretrained(
        args.model_path,
        **load_kwargs
    )
    model.eval()
    if "device_map" not in load_kwargs:  # if not using device_map, move to GPU
        model.to(args.device)

    # Initialize vision modules if present
    if hasattr(model.get_model(), "initialize_vision_modules"):
        print("[INFO] Initializing vision modules...")
        class DummyVisionArgs:
            vision_tower = args.model_path
            mm_vision_select_layer = -1
            mm_vision_select_feature = "patch"
            mm_patch_merge_type = "flat"
            mm_use_im_start_end = False
            mm_use_im_patch_token = True
            pretrain_mm_mlp_adapter = None
            tune_mm_mlp_adapter = False
            freeze_mm_mlp_adapter = False

        dummy_args = DummyVisionArgs()
        model.get_model().initialize_vision_modules(model_args=dummy_args, fsdp=None)

        # Move vision tower to GPU if needed
        if hasattr(model, "get_vision_tower"):
            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.float16, device=args.device)

    # Attach saliency wrapper
    print("[INFO] Attaching SalienceSingleImageWrapper...")
    sal_wrapper = SalienceSingleImageWrapper(
        config_file="/scratch/jl9356/salience_llava/QAGNet_main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        resume=False,
        opts=[]
    )
    model.set_sal_wrapper(sal_wrapper)

    # Optionally load & merge LoRA
    if args.lora_path:
        print(f"[INFO] Loading LoRA from {args.lora_path}")
        lora_model = PeftModel.from_pretrained(model, args.lora_path)
        model = lora_model.merge_and_unload()
        model.eval()
        if "device_map" not in load_kwargs:
            model.to(args.device)

    # Build tokenizer
    print("[INFO] Building tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Determine conversation template
    if "llama-2" in args.model_path.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in args.model_path.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in args.model_path.lower():
        conv_mode = "chatml_direct"
    elif "v1" in args.model_path.lower():
        conv_mode = "llava_v1"
    elif "mpt" in args.model_path.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode:
        conv_mode = args.conv_mode

    conv_template = conv_templates[conv_mode].copy()
    roles = conv_template.roles

    # -------------------------------------------------------------------------
    # 2c. Read the annotations JSON
    # -------------------------------------------------------------------------
    data = load_annotations(args.annotations_json)
    print(f"[INFO] Loaded {len(data)} annotation items.")

    # -------------------------------------------------------------------------
    # 2d. Generate predictions (one-by-one)
    # -------------------------------------------------------------------------
    # We'll wrap our for-loop with tqdm, and also print debug info each iteration.
    for idx, item in enumerate(tqdm(data, desc="Evaluating images", total=len(data))):
        image_path = item["image"]
        if not os.path.exists(image_path):
            tqdm.write(f"[WARNING] Image not found: {image_path}")
            continue

        # Extract ground-truth
        gt_caption = get_ground_truth_caption(item.get("conversations", []))
        if not gt_caption:
            continue
        all_references.append(gt_caption)

        # Build a single user prompt
        user_text = args.user_prompt.strip()

        # Insert <Image> token(s) if needed
        if getattr(model.config, "mm_use_im_start_end", False):
            user_text = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{user_text}"
        else:
            user_text = f"{DEFAULT_IMAGE_TOKEN}\n{user_text}"

        # Build conversation
        conv = conv_template.copy()
        conv.append_message(roles[0], user_text)
        conv.append_message(roles[1], None)
        prompt = conv.get_prompt()

        # Load/process image
        image = load_image(image_path)
        vision_tower = getattr(model, "get_vision_tower", lambda: None)()
        image_processor = getattr(vision_tower, "image_processor", None)
        image_tensor = process_images([image], image_processor, model.config)

        if isinstance(image_tensor, list):
            image_tensor = [img.to(args.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(args.device, dtype=torch.float16)

        # Tokenize prompt
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(args.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=False,
            )

        # Decode text
        pred_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        all_predictions.append(pred_caption)

        # Print debug info: which image, predicted text, etc.
        # We use tqdm.write so that it doesn't break the progress bar.
        tqdm.write(f"[{idx+1}/{len(data)}] Image: {image_path}")
        tqdm.write(f"Prediction: {pred_caption}\n")

    # -------------------------------------------------------------------------
    # 2e. Compute metrics
    # -------------------------------------------------------------------------
    print("\n[INFO] Computing ROUGE ...")
    rouge_scores = rouge_metric.compute(predictions=all_predictions, references=all_references)

    print("[INFO] Computing METEOR ...")
    meteor_scores = meteor_metric.compute(predictions=all_predictions, references=all_references)

    print("\n================ EVALUATION RESULTS ================\n")
    print("ROUGE Scores:")
    for k, v in rouge_scores.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nMETEOR: {meteor_scores['meteor']:.4f}\n")


###############################################################################
# 3. Entry Point
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the base LLaVA/saliency checkpoint.")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to a LoRA checkpoint folder (if any).")
    parser.add_argument("--annotations-json", type=str, required=True,
                        help="Path to the JSON file containing images + GT captions.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--user-prompt", type=str,
                        default="Please describe the image in detail.",
                        help="Prompt text for your caption request.")
    args = parser.parse_args()
    main(args)
