# NLP-to-LLMs
Set of small python exercises to learn about NLP standard techniques up to current LLMs methods.

## Named Entity Recognition

Named Entity Recognition is a subtask of information extraction. It aims at **locating** and **classifying** named entities in the text. As all classification methods it required pre-defined categories: person names, organizations, locations, monetary values, etc. To correctly classify such instances in the text the algorithm needs access to a subsequence of text, a temporal window, since one single word does not provide enough context.

The exercises provide two ways to address the problem, using the same dataset. At the beginning both pipelines read the .csv file and construct a dataset including the split between training and testing. 

### NER using CRF + Limited Memory BFGS

Limited Memory BFGS is a computational efficient algorithm popular for parameters estimation. It is an "ancient" machine learning technique. The main advantage of it is that instead of keeping the Hessian matrix H to optimize the variables, it maintains past updates and gradients (over a short time span) to implicitly perform operations that qould usually require building the hessian matrix. 

### NER using a Bidirectional LSTM  

The LSTM is the Long Short-Term Memory  unit, the BiLSTM is a type of Recurrent Neural Network that not only uses these units, but process data in both directions (that's why is called bidirectional). The double direction allows capturing dependencies in both past and future words, obtaining a richer context. 


