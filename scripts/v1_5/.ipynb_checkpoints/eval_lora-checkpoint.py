#!/usr/bin/env python
# eval_lora.py

import argparse
import torch
import requests
from io import BytesIO
from PIL import Image

# We need a direct transformers import here
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


def load_image(image_file):
    """Load an image from a local path or a URL."""
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # 1) Turn off lazy init
    disable_torch_init()

    # 2) Decide whether to load model in 16-bit, 8-bit, etc.
    load_kwargs = {}
    if args.load_4bit or args.load_8bit:
        # Example bitsandbytes config if needed
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

    # 3) Load our saliency-enabled LLaVA model
    print(f"[INFO] Loading MySaliencyLlamaForCausalLM from: {args.model_path}")
    model = MySaliencyLlamaForCausalLM.from_pretrained(
        args.model_path,
        **load_kwargs
    )
    model.eval()
    if "device_map" not in load_kwargs:  # if not using a device_map, move to GPU
        model.to(args.device)

    # 4) Initialize vision tower if present
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

        dummy = DummyVisionArgs()
        model.get_model().initialize_vision_modules(model_args=dummy, fsdp=None)

        # Move vision tower to GPU if needed
        if hasattr(model, "get_vision_tower"):
            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.float16, device=args.device)

    # 5) Attach saliency wrapper so it doesn't remain None
    print("[INFO] Attaching SalienceSingleImageWrapper...")
    sal_wrapper = SalienceSingleImageWrapper(
        config_file="/scratch/jl9356/salience_llava/QAGNet_main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        resume=False,
        opts=[]
    )
    model.set_sal_wrapper(sal_wrapper)  # So encode_images sees the real wrapper

    # 6) Load & merge LoRA, if provided
    if args.lora_path:
        print(f"[INFO] Loading LoRA from {args.lora_path}")
        lora_model = PeftModel.from_pretrained(model, args.lora_path)
        model = lora_model.merge_and_unload()
        model.eval()
        if "device_map" not in load_kwargs:
            model.to(args.device)

    # 7) Build tokenizer
    #    If you do the same as train.py:
    print("[INFO] Building tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token

    # 8) Determine conversation template
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

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # 9) Load image, *this time with a real image_processor*
    image = load_image(args.image_file)
    # Retrieve image_processor from vision tower
    vision_tower = model.get_vision_tower() if hasattr(model, "get_vision_tower") else None
    image_processor = getattr(vision_tower, "image_processor", None)

    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(args.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(args.device, dtype=torch.float16)

    # 10) Interactive chat loop
    print("[INFO] Enter a query (press Enter on empty line to exit).")
    first_round = True
    while True:
        user_text = input(f"{roles[0]}: ")
        if not user_text.strip():
            print("Exiting.")
            break

        print(f"{roles[1]}: ", end="", flush=True)

        # Insert <Image> token on the first user prompt
        if first_round:
            first_round = False
            if getattr(model.config, "mm_use_im_start_end", False):
                user_text = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n" + user_text
            else:
                user_text = f"{DEFAULT_IMAGE_TOKEN}\n{user_text}"

        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(args.device)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=False,
            )

        outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n[DEBUG] prompt:", prompt)
            print("[DEBUG] outputs:", outputs, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the base LLaVA/saliency checkpoint.")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to a LoRA checkpoint folder (if any).")
    parser.add_argument("--image-file", type=str, required=True,
                        help="Image path/URL.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
