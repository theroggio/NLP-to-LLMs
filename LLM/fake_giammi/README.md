## STEPS

1. Start with pre-trained model, like 'GroNLP/gpt2-small-italian'
2. Create personal dataset as
```
{
	"prompt": "USER: domanda?\nYOU: altre cose",
	"response": "esempio di risposta come il prompt" 
}
```
3. Use PEFT (Parameter-Efficient Fine-Tuning) to fine tune on new dataset
4. Tune for instruction type if needed
5. Test and iterate


## Pipeline loader

```
from transformers import AutoTokenizer, AutoModel, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("GroNLP/gpt2-small-italian")
model = AutoModel.from_pretrained("GroNLP/gpt2-small-italian")  # PyTorch

```

## Set up PEFT

```
from peft import LoraConfig, TaskType
from peft import get_peft_model

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
```

you can check how many parameters you are really training running

```
model.print_trainable_parameters()
```

## Training for real

```
args = transformers.TrainingArguments(
    output_dir="./out/folder/you/like", 
    learning_rate=1e-3, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    num_train_epochs=2, 
    weight_decay=0.01, 
    eval_strategy="epoch", 
    save_strategy="epoch", 
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## Save (local and online)

```
model.save_pretrained("out/name")

from huggingface_hub import notebook_login
notebook_login()
model.push_to_hub("path/to/hugging/hub")
```


## Inference

```
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

# if on hugging face
model = AutoPeftModelForCausalLM.from_pretrained("path/to/hugging/model")

# if local it is a mess
config = your_petf_config
base_model = AutoModelForCausalLM.from_pretrained("base/model/you/started/from")
model = PeftModel.from_pretrained(base_model, "loca/path/to/model")

tokenizer = AutoTokenizer.from_pretrained("tokenizer/name")

model = model.to("cuda")
model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
```

## Samples (example of samples)

{
  "prompt": "USER: Ehi Gianmarco, che fai di bello stasera?\nASSISTANT:",
  "response": "Pensavo di starmene tranquillo con un po’ di musica e magari una serie. Hai qualche consiglio?"
} 

{
  "prompt": "USER: Ho avuto una giornata assurda... il capo fuori di testa.\nASSISTANT:",
  "response": "Uff, ti capisco. Respira, stacca un po’. Vuoi rantare o preferisci distrarti?"
}

{
  "prompt": "USER: Non riesco a decidere tra lasagna o sushi.\nASSISTANT:",
  "response": "Wow, guerra fra mondi. Ti senti più coccole italiane o avventura giapponese? 😄"
}

{
  "prompt": "USER: Mi sa che mollo tutto e parto per Bali.\nASSISTANT:",
  "response": "Faccio le valigie sono pronto in 3 minuti."
}
