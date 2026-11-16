
import json
from pathlib import Path

from datasets import Dataset

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    losses,
)


BASE_MODEL = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

OUTPUT_DIR = "artifacts/pairwise_complementarity_v1"


def load_training_data():
    path = Path("data/training/train_pairs.json")

    records = json.loads(
        path.read_text(encoding="utf-8")
    )

    return Dataset.from_list([
        {
            "sentence1": record["cart_product"],
            "sentence2": record["candidate"],
            "label": float(record["label"]),
        }
        for record in records
    ])


def main():
    dataset = load_training_data()

    print(f"Training examples: {len(dataset)}")

    # Initialize a single-output cross-encoder.
    model = CrossEncoder(
        BASE_MODEL,
        num_labels=1,
        max_length=192,
    )

    args = CrossEncoderTrainingArguments(
        output_dir="artifacts/training_checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=losses.BinaryCrossEntropyLoss(model),
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)

    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
