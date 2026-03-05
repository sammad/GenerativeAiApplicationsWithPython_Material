#%% packages
import logging
from pprint import pprint
# TODO: import the necessary packages
from attr import dataclass
from transformers import pipeline
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
#%% model selection
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentimentalAnalysis = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")
#%% data
feedback = [
    "I recently bought the EcoSmart Kettle, and while I love its design, the heating element broke after just two weeks. Customer service was friendly, but I had to wait over a week for a response. It's frustrating, especially given the high price I paid.",
    "Die Lieferung war super schnell, und die Verpackung war großartig! Die Galaxy Wireless Headphones kamen in perfektem Zustand an. Ich benutze sie jetzt seit einer Woche, und die Klangqualität ist erstaunlich. Vielen Dank für ein tolles Einkaufserlebnis!",
    "Je ne suis pas satisfait de la dernière mise à jour de l'application EasyHome. L'interface est devenue encombrée et le chargement des pages prend plus de temps. J'utilise cette application quotidiennement et cela affecte ma productivité. J'espère que ces problèmes seront bientôt résolus."
]

# %% function
@dataclass
class Feedback:
    description: str
    sentiment: float
    label: str
# TODO: define the function process_feedback
def process_feedback(feedback_list):
    sentiments=sentimentalAnalysis(feedback_list);
    logging.info("Sentiment analysis results: %s", sentiments)
    classifier_results=classifier(feedback_list, candidate_labels=["defect", "delivery", "interface"])
    feedback_objects = []
    for i in range(len(feedback_list)):
        feedback_obj = Feedback(description=feedback_list[i], sentiment=sentiments[i]["label"], label=classifier_results[i]["labels"][0])
        feedback_objects.append(feedback_obj)
        logging.info("Feedback object created: %s", feedback_obj)
    return feedback_objects

feedbacks=process_feedback(feedback)
logging.info("Processed feedback: %s", feedbacks)
#%% Test
# TODO: test the function process_feedback
test_feedback = ["This is a positive review!", "This is a negative review."]
test_results = process_feedback(test_feedback)
logging.info("Test results: %s", test_results)