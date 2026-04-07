from langchain_ollama import OllamaLLM
#%%
llm = OllamaLLM(model="gemma2:2b")
response = llm.invoke("What is an LLM?")
response
#%%