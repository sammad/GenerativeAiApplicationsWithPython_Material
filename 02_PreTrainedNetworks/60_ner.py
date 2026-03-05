#%%
import logging
from pprint import pprint

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

#%% model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

logging.info("Tokenizer loaded: %s", tokenizer)
logging.info("Tokenizer class: %s", tokenizer.__class__.__name__)

example = "The dog Perky was roaming in teh roads of Palampur"
enc = tokenizer(
    example,
    return_tensors="pt",
    return_offsets_mapping=True,
    truncation=True,
)

input_ids = enc["input_ids"][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

logging.info("Input IDs: %s", input_ids)
logging.info("Tokens: %s", tokens)
logging.info("Attention mask: %s", enc["attention_mask"][0].tolist())
logging.info("Offsets: %s", enc["offset_mapping"][0].tolist())

unk_id = tokenizer.unk_token_id
unk_positions = [i for i, token_id in enumerate(input_ids) if token_id == unk_id]
unk_tokens = [tokens[i] for i in unk_positions]

logging.info("UNK token id: %s", unk_id)
logging.info("UNK positions: %s", unk_positions)
logging.info("UNK tokens at those positions: %s", unk_tokens)

#%% pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
ner_results = nlp(example)
pprint(ner_results)
# %%
