#%% packages
import os
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# %%
# Model overview: https://console.groq.com/docs/models
MODEL_NAME = 'llama-3.3-70b-versatile'
model = ChatGroq(model_name=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('GROQ_API_KEY'),
                   max_tokens=50)

# %% Run the model
prompt = "What is a Huggingface?"
logging.info(f"Sending prompt to LLM: {prompt}")

# Create internal message structure and log it
messages = [HumanMessage(content=prompt)]
logging.info(f"Internal prompt structure: {messages}")

res = model.invoke(messages)
# %% find out what is in the result
res.model_dump()
# %% only print content
print(res.content)
# %%
