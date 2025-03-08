# train_mem_MLP.py
# Example: Single-MLP bridging for LLaVA, with DeepSpeed zero-3 & custom LR.

import os
import json
import torch
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Dict

import transformers
from transformers import (
    AutoTokenizer,
    CLIPVisionModel,
    CLIPProcessor,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from PIL import Image

IGNORE_INDEX = -100

# --------------------------------------------------------------------------
# 1. Subclass: single MLP bridging
# --------------------------------------------------------------------------
class MySimpleLlamaForCausalLM(transformers.LlamaForCausalLM):
    """
    Subclass of LlamaForCausalLM that:
      - Adds a single MLP (mm_projector) for vision features.
      - Overrides forward(...) to accept optional `images`.
    """
    def __init__(self, config):
        super().__init__(config)

        # If you want 2-layer MLP with gelu, you can modify here:
        # Example for a 2-layer with GELU:
        # from torch import nn
        # hidden_sz = config.hidden_size
        # self.mm_projector = nn.Sequential(
        #     nn.Linear(getattr(config, "vision_hidden_size", 768), hidden_sz),
        #     nn.GELU(),
        #     nn.Linear(hidden_sz, hidden_sz)
        # )
        # For a single-layer, just do:
        vision_hidden_size = getattr(config, "vision_hidden_size", 768)
        self.mm_projector = torch.nn.Linear(vision_hidden_size, config.hidden_size)

        self.vision_model = None
        self._vision_processor = None

    def initialize_vision_modules(self, model_args, fsdp=None):
        # Optionally do something here. We'll do lazy load in forward()
        pass

    def get_vision_tower(self):
        """Create/return a CLIPVisionModel from self.config.vision_tower."""
        vt = getattr(self.config, "vision_tower", None)
        if vt is None:
            raise ValueError("vision_tower not specified in config.")
        return CLIPVisionModel.from_pretrained(vt)

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        pass

    def forward(self, input_ids=None, attention_mask=None, images=None, **kwargs):
        if images is not None:
            if self.vision_model is None:
                vt = getattr(self.config, "vision_tower", None)
                if vt is None:
                    raise ValueError("vision_tower not specified in config.")
                self.vision_model = CLIPVisionModel.from_pretrained(vt).to(input_ids.device)

            # forward pass of CLIP
            vision_outputs = self.vision_model(pixel_values=images)
            if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
                vision_features = vision_outputs.pooler_output  # (B, hidden_dim)
            else:
                # fallback
                vision_features = vision_outputs.last_hidden_state.mean(dim=1)

            # project
            projected = self.mm_projector(vision_features)    # (B, hidden_dim)
            projected = projected.unsqueeze(1)               # (B, 1, hidden_dim)

            text_embeds = self.get_input_embeddings()(input_ids)  # (B, seq_len, hidden_dim)
            new_embeds = torch.cat([projected, text_embeds], dim=1)

            if attention_mask is not None:
                batch_size = attention_mask.size(0)
                extra_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([extra_mask, attention_mask], dim=1)

            return super().forward(inputs_embeds=new_embeds, attention_mask=attention_mask, **kwargs)
        else:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


# --------------------------------------------------------------------------
# 2. Arg classes
# --------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="")
    version: str = field(default="v1")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_projector_type: str = field(default='linear')
    mm_vision_select_layer: int = field(default=-1)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = None


@dataclass
class DataArguments:
    data_path: str = field(default="")
    lazy_preprocess: bool = field(default=False)
    is_multimodal: bool = field(default=False)
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class MyTrainingArguments(TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    mm_projector_lr: Optional[float] = field(default=None)
    group_by_modality_length: bool = field(default=False)


# --------------------------------------------------------------------------
# 3. Minimal dataset
# --------------------------------------------------------------------------
class LazySupervisedDataset(Dataset):
    """
    Example dataset that reads a JSON list of items, each possibly with "image".
    If there's "image", we store the path to be loaded in __getitem__.
    """
    def __init__(self, data_path, tokenizer, data_args: DataArguments):
        super().__init__()
        with open(data_path, "r") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        # If there's an image, load a PIL
        if "image" in data and data["image"]:
            image_path = os.path.join(self.data_args.image_folder, data["image"])
            img = Image.open(image_path).convert("RGB")
            data["pil_image"] = img
        return data


# --------------------------------------------------------------------------
# 4. DataCollator
# --------------------------------------------------------------------------
class DataCollator:
    """
    Simple collator: tokenizes the text, processes images via CLIPProcessor, etc.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vision_processor = None

    def __call__(self, batch):
        # gather text
        texts = []
        for item in batch:
            if "conversations" in item:
                # Possibly multi-turn in your real code. We do a simple single-turn example:
                # E.g. item["conversations"] = [ { "from":"human", "value":"Hi" }, {"from":"assistant",...} ]
                # We'll just pick the entire conversation's text or the first chunk
                texts.append(item["conversations"][0]["value"])
            elif "text" in item:
                texts.append(item["text"])
            else:
                # fallback
                texts.append("")

        # tokenize
        encoded = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        attention_mask = encoded["attention_mask"]

        # images
        images_list = []
        for item in batch:
            if "pil_image" in item:
                if self.vision_processor is None:
                    # Adjust if your tower is openai/clip-vit-large-patch14-336 or something else
                    self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
                pixel_out = self.vision_processor(images=item["pil_image"], return_tensors="pt")
                images_list.append(pixel_out["pixel_values"][0])
            else:
                images_list.append(None)

        # attempt to stack
        if any(img is not None for img in images_list):
            shapes = [img.shape for img in images_list if img is not None]
            if len(shapes) > 0 and all(s == shapes[0] for s in shapes):
                new_list = []
                for img in images_list:
                    if img is None:
                        new_list.append(torch.zeros_like(images_list[0]))
                    else:
                        new_list.append(img)
                images_tensor = torch.stack(new_list, dim=0)
            else:
                images_tensor = images_list
        else:
            images_tensor = None

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images_tensor,
        }


# --------------------------------------------------------------------------
# 5. Safe Save
# --------------------------------------------------------------------------
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    # If using deepspeed or fsdp, do it carefully. Otherwise just:
    if trainer.deepspeed or trainer.args.fsdp:
        trainer.save_model(output_dir)
    else:
        trainer.save_model(output_dir)


# --------------------------------------------------------------------------
# 6. Main train function
# --------------------------------------------------------------------------
def train(attn_implementation=None):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1) Load the bridging model (or fallback LlamaForCausalLM)
    compute_dtype = (
        torch.float16 if training_args.fp16 else
        (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if model_args.vision_tower:
        # Use bridging model
        model = MySimpleLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
        )
        model.config.vision_tower = model_args.vision_tower
    else:
        # fallback
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=compute_dtype,
        )

    # 2) Possibly freeze backbone
    if model_args.freeze_backbone:
        # freeze everything in LLaMA
        model.model.requires_grad_(False)
        # unfreeze only bridging MLP
        if hasattr(model, "mm_projector"):
            for p in model.mm_projector.parameters():
                p.requires_grad = True

    # 3) Possibly apply LoRA
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # Typically: target_modules=["q_proj", "k_proj", "v_proj", ...]
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    # 4) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    # 5) Vision modules if needed
    if model_args.vision_tower:
        model.initialize_vision_modules(model_args, fsdp=training_args.fsdp)
        model.initialize_vision_tokenizer(model_args, tokenizer)

    # 6) Build dataset/collator
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    data_collator = DataCollator(tokenizer=tokenizer)

    # 7) Build trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 8) If mm_projector_lr is given, override create_optimizer/create_scheduler
    if training_args.mm_projector_lr is not None:
        from transformers.optimization import AdamW, get_scheduler

        # separate param groups: bridging vs. everything else
        def param_groups(model):
            proj_params = []
            base_params = []
            for n, p in model.named_parameters():
                if "mm_projector" in n and p.requires_grad:
                    proj_params.append(p)
                else:
                    # also includes LoRA if lora_enable==True
                    base_params.append(p)
            return [
                {"params": proj_params, "lr": training_args.mm_projector_lr},
                {"params": base_params, "lr": training_args.learning_rate},
            ]

        def custom_create_optimizer(self):
            pg = param_groups(self.model)
            self.optimizer = AdamW(
                pg,
                lr=self.args.learning_rate,  # global LR for the base group
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer

        def custom_create_scheduler(self, num_training_steps: int):
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            return self.lr_scheduler

        trainer.create_optimizer = custom_create_optimizer.__get__(trainer, Trainer)
        trainer.create_scheduler = custom_create_scheduler.__get__(trainer, Trainer)

    # 9) Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # 10) Save final
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)


if __name__ == "__main__":
    train()
