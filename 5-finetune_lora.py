"""Fine-tune a small open model on Darija instructions with LoRA.

Targets a single 12 GB GPU (RTX 3500 Ada, RTX 3060 12GB, ...): 4-bit
quantized base model + LoRA adapters.

Usage:
    pip install -r requirements-finetune.txt
    python finetune_lora.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --dataset sample_instruct.jsonl \
        --output ./darija-lora

The dataset is JSONL with {"instruction": ..., "input": ..., "output": ...}
per line. ``input`` may be an empty string.

After training, load the adapter on the base model with PEFT, or merge and
convert to GGUF for Ollama/llama.cpp.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

PROMPT_TEMPLATE = (
    "### Instruction (Darija or Arabic):\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def load_jsonl_dataset(path: str):
    from datasets import Dataset  # deferred import: heavy dependency

    rows: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for row in rows:
        row["text"] = PROMPT_TEMPLATE.format(
            instruction=row["instruction"],
            input=row.get("input", ""),
            output=row["output"],
        )
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="sample_instruct.jsonl")
    parser.add_argument("--output", default="./darija-lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = load_jsonl_dataset(args.dataset)

    def tokenize(row):
        tokens = tokenizer(
            row["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokens["labels"] = list(tokens["input_ids"])
        return tokens

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    training = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
