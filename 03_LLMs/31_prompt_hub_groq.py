#%% packages
import os
from langchainhub import Client
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import loads
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pprint import pprint

#%% fetch prompt
hub = Client()
raw_prompt = hub.pull("hardkothari/prompt-maker")
prompt = loads(raw_prompt) if isinstance(raw_prompt, str) else raw_prompt

#%% get input variables
pprint(prompt.input_variables)

# %% model
MODEL_NAME = "llama-3.3-70b-versatile"
model = ChatGroq(
    model=MODEL_NAME,
    temperature=0.5,  # controls creativity
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=50,
)

# %% chain
chain = prompt | model | StrOutputParser()

# %% invoke chain
lazy_prompt = "summer, vacation, beach"
task = "Shakespeare poem"
improved_prompt = chain.invoke({"lazy_prompt": lazy_prompt, "task": task})
# %%
print(improved_prompt)

# %% run model with improved prompt
res = model.invoke(improved_prompt)
print(res.content)

# %%
res = model.invoke("summer, vacation, beach, Shakespeare poem")
print(res.content)
# %%
