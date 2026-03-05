#%% packages
from transformers import pipeline
from transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens({'mask_token': '[MASK]'})
unmasker = pipeline(task='fill-mask', model='bert-base-uncased', tokenizer=tokenizer)
#%%
unmasker("You are a very [MASK] person")
# %%
