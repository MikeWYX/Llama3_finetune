MAX_LEN = 512 
checkpoint = "/data/wuyux/Llama-3-8B-Instruct"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"

from datasets import load_dataset
dataset = load_dataset("mehdiiraqui/twitter_disaster")

from datasets import Dataset
data = dataset['train'].train_test_split(train_size=0.8, seed=42)
data['val'] = data.pop("test")
# Convert the test dataframe to HuggingFace dataset and add it into the first dataset
data['test'] = dataset['test']

import pandas as pd
pos_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().target.value_counts()[1])
neg_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().target.value_counts()[0])

# Load Llama 2 Tokenizer
from transformers import AutoTokenizer, DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

def llama_preprocessing_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

col_to_delete = ['id', 'keyword','location', 'text']
tokenized_datasets = data.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
tokenized_datasets = tokenized_datasets.rename_column("target", "label")
tokenized_datasets.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification
import torch
model =  AutoModelForSequenceClassification.from_pretrained(
  pretrained_model_name_or_path=checkpoint,
  num_labels=2,
  device_map="auto",
  offload_folder="offload",
  trust_remote_code=True
)
model.config.pad_token_id = model.config.eos_token_id

from peft import get_peft_model, LoraConfig, TaskType
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none", 
    target_modules=[
        "q_proj",
        "v_proj",  
    ],
)

model = get_peft_model(model, peft_config)

import evaluate
import numpy as np

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_pred 
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    return {'accuracy': accuracy}

from transformers import Trainer

class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

from transformers import TrainingArguments, Trainer, TrainerCallback

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

lr = 1e-4
batch_size = 8
num_epochs = 30
training_args = TrainingArguments(
    output_dir="llama-lora-token-classification",
    learning_rate=lr,
    lr_scheduler_type= "constant",
    warmup_ratio= 0.1,
    max_grad_norm= 0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    gradient_checkpointing=True,
)

trainer = WeightedCELossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

class LogCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])

log_callback = LogCallback()

def evaluate_model(trainer, dataset):
    trainer.model.eval()
    eval_results = trainer.evaluate(eval_dataset=dataset)
    accuracy = eval_results["eval_accuracy"]
    return accuracy


pre_train_accuracy = evaluate_model(trainer, tokenized_datasets["val"])
print(f"Accuracy before fine-tuning: {pre_train_accuracy:.4f}")

trainer.add_callback(log_callback)
trainer.train()

post_train_accuracy = evaluate_model(trainer, tokenized_datasets["val"])
print(f"Accuracy after fine-tuning: {post_train_accuracy:.4f}")

import matplotlib.pyplot as plt
plt.plot(log_callback.losses)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("loss.png")