# RAG

The main.py file is complete, but it needs a local model to run. 

Open a second terminal, and run 

```
curl -fsSL https://ollama.com/install.sh | sh
```

to install ollama, and then 

```
ollama run llama2
```

to run it. 

Then on the first terminal run ```python main.py``` to see the results using all free models, no API required. It will print both results from our rag and from the "built-in" rag procedure og langchain.
