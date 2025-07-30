# data import
import pandas as pd
import random
import numpy as np
from datasets import Dataset

# llm import
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TFAutoModel
from peft import LoraConfig, TaskType, get_peft_model


# load jsonl file
data = np.array(pd.read_json("giammi.jsonl", lines=True))

# create random indexing for shuffling data
idx_random = np.arange(0, data.shape[0])
random.shuffle(idx_random)

# shuffle data
data = data[idx_random]

# split train and val 
ratio = 0.8
nelem = int(data.shape[0] * ratio)
train_data = data[0:nelem]
val_data = data[nelem:]

# transform data to tokens, important to define pad and truncation!
tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
train_tokens = tokenizer(train_data.tolist(), padding=True, truncation=True, max_length=50)
val_tokens = tokenizer(val_data.tolist(), padding=True, truncation=True, max_length=50)

# need to define labels
train_tokens["labels"] = train_tokens["input_ids"].copy()
val_tokens["labels"] = val_tokens["input_ids"].copy()


# define datasts
train_dl = Dataset.from_dict(train_tokens)
val_dl = Dataset.from_dict(val_tokens)

# define model 
model = AutoModelForCausalLM.from_pretrained("GroNLP/gpt2-small-italian")  # PyTorch

# set up PFET
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# set up training args and trainer
args = transformers.TrainingArguments(
    output_dir="./out/test",
    learning_rate=1e-3,
    per_device_train_batch_size=96,
    per_device_eval_batch_size=96,
    num_train_epochs=200,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)


trainer = transformers.Trainer(
    model=model,
    args=args,
    train_dataset=train_dl,
    eval_dataset=val_dl,
    tokenizer=tokenizer,
    #data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

trainer.train()

##### inference
model.eval()


print("Eval yourself. Leave blank to close.")
while True:
    question = input("")
    if question == "":
        break
    # make question as template
    real_question = f"USER: {question}\nASSISTANT:"
    inputs = tokenizer([real_question], padding=True, truncation=True, max_length=50, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

save = input("Press 's' to save the model. anything else will kill the process.")
if save == 's':
    model.save_pretrained("./out/model")






