#!/usr/bin/env python3
import argparse
import os
import json
import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.utils import disable_torch_init

from PIL import Image
from transformers import TextStreamer

# For loading LoRA weights
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
    print("Warning: peft is not installed. If you need LoRA support, please install it with `pip install peft`.")

def load_image(image_file):
    """
    Loads an image from a local path or URL.
    """
    if image_file.startswith("http://") or image_file.startswith("https://"):
        import requests
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def main(args):
    # Disable Torch lazy initialization
    disable_torch_init()
    print("Starting inference...")

    # Load the base model, tokenizer, and image processor
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device
    )

    # If LoRA path is provided, load the LoRA adapter
    if args.lora_path is not None:
        if PeftModel is None:
            raise ImportError("peft is not installed. Install with `pip install peft` to use LoRA models.")
        print(f"Loading LoRA adapter from {args.lora_path} ...")
        # Load the LoRA adapter on top of the base model
        model = PeftModel.from_pretrained(
            model,
            args.lora_path,
            torch_dtype=torch.float16 if args.device == "cuda" else None
        )

    model.eval().to(args.device)

    # Gather images from the specified directory
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_list = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if os.path.splitext(f.lower())[1] in valid_ext
    ])
    if not image_list:
        print(f"No valid images found in {args.image_dir}")
        return

    if args.max_images > 0 and len(image_list) > args.max_images:
        print(f"Found {len(image_list)} images. Only processing the first {args.max_images}.")
        image_list = image_list[:args.max_images]

    results = []
    
    # Fixed human prompt for JSON record (example)
    fixed_human_prompt = (
        "Please describe the image in the order humans typically find most important, "
        "starting with the elements that are most salient, then moving on to the less prominent details."
    )

    # Process each image
    for idx, img_path in enumerate(image_list):
        print(f"\nProcessing image {idx + 1}/{len(image_list)}: {img_path}")

        # Load the image and get its size
        image = load_image(img_path)
        image_size = image.size

        # Preprocess the image
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0].unsqueeze(0).to(model.device, dtype=torch.float16)
        else:
            image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)

        # Ensure <image> token is present in the prompt
        gen_prompt = args.prompt.strip()
        if "<image>" not in gen_prompt:
            gen_prompt += "\n<image>"

        # Tokenize the prompt (the helper function replaces the <image> token appropriately)
        input_ids = tokenizer_image_token(
            gen_prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        # Create a streamer to print the generated text as it is produced
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Generate the model's response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True
            )

        outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"\nModel answer for image '{img_path}':\n{outputs}")

        if args.debug:
            print("\n=== DEBUG INFO ===")
            print("Generation Prompt:\n", gen_prompt)
            print("Output:\n", outputs)
            print("=== END DEBUG ===\n")

        # Build a JSON record
        record = {
            "id": os.path.splitext(os.path.basename(img_path))[0],
            "image": img_path,
            "conversations": [
                {"from": "human", "value": fixed_human_prompt},
                {"from": "gpt", "value": outputs}
            ]
        }
        results.append(record)

    # Save all records to a JSON file
    output_path = "record_vizwiz_salience_v0.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nAll records saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the pretrained base model (e.g., Hugging Face repo or local path).")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Optional base model identifier if needed.")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to the LoRA checkpoint directory.")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing images to process.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="The prompt to send to the model for generation.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1,
                        help="Limit how many images to process (<=0 means all).")

    args = parser.parse_args()
    main(args)
