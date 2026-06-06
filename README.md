### Project for examination preparation
This project is at the moment a prototype for the logic needed for exam preparation. Infuture itwill be a Python GUI / HTML 5  Applicationfor research  end exam preparation. 
I enhanced the project to use a own chat model, because the chat models are expensive for private developement.
The application has a ASAG logic for question / answer scoring. RAG for retrieval of information as well as a hybrid search component.

#### Key Features

- Research via _**RAG**_ on the vector databases. The databases are sorted by discipline.
- Research on documents via Hybrid search. 
Transformer search and Fulltext Search. Both are put together 
by a ranking and the documents which have the highest rank in both searches are 
used as top matches.
- A chat functionality like in _**ChatGPT**_.
- A _**ASAG**_ scoring with different scores is used to score answers in a test.
- A possibility to create examinations and execute them as test examination or final examination.
- A group chat feature.