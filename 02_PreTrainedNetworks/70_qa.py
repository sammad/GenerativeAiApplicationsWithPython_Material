#%% packages
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pprint import pprint
#%% constants
MODEL = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline(task='question-answering', model=MODEL, tokenizer=MODEL)
QA_input = {
    'question': 'What are the downsides of remote work?',
    'context': 'It reduces the interpersonal communication and collaboration that is essential for innovation and creativity. It can lead to feelings of isolation and loneliness, which can negatively impact mental health. It can also make it difficult to establish and maintain a strong company culture, which can lead to decreased employee engagement and retention.'
}
res = nlp(QA_input)
pprint(res)

# %%
