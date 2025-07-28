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
