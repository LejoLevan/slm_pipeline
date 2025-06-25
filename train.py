from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from preprocess import preprocess
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

model = AutoModelForQuestionAnswering.from_pretrained("models/tinyroberta-squad2")
model_name = "tinyroberta"

excluded_keywords = set()
for name, module in model.named_children():
    if isinstance(module, nn.Linear):
        excluded_keywords.add(name)

linear_layer_names = set()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        if not any(exclude in name for exclude in excluded_keywords):
            last_name = name.split('.')[-1]
            linear_layer_names.add(last_name)

#explain what is lora, why I am using it (can put in readme)
#put highlevel
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['query', 'value'],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.QUESTION_ANS
)

model = get_peft_model(model, lora_config)

args = TrainingArguments(
    "tinyroberta-finetuned-squad",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

tokenizer = AutoTokenizer.from_pretrained("models/tinyroberta-squad2")
tokenized_datasets = preprocess()

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("models/trained")
trainer.save_pretrained(f"models/trained/{model_name}")