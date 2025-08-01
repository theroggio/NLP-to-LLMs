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


# FastAPI example

Run

```
uvicorn main:app --reload
```

to start the 'web api', you can then load ```http://127.0.0.1:8000/docs ``` to run single requests. There is no web page, since it is not the main purpose of this repo. However, after getting the respose via the HTTP requests, one can display them and do whatever with them.
