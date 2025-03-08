import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


from datasets import load_dataset


model_name = "liuhaotian/LLaVA-7B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)



def main():
    model_name = "liuhaotian/LLaVA-7B"
    tokenizer = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map = "auto",
    )

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    training_args = TrainingArguments(
        output_dir="./llava-finetuned-checkpoint",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_steps=200,
        save_total_limit=1,
        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    for name, module in model.named_modules():
        if "attn" in name.lower():
            print(name, module)
if __name__ == "__main__":
    main()