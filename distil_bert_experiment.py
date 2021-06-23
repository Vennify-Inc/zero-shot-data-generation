"""
Explores using a zero-shot distilbert text classification model
By Eric Fillion @ Vennify.ai
"""
from transformers import pipeline
from datasets import load_dataset

from util import *


def distil_bert_experiment():

    task = "zero-shot-classification"
    zero_shot_model = "typeform/distilbert-base-uncased-mnli"

    print("Loading a zero-shot transformer model...")
    zero_shot_classifier = pipeline(task, zero_shot_model)
    zero_shot_labels = ["negative", "positive"]

    print("\nFetching data from Hugging Face's dataset distribution network...")

    eval_data = load_dataset('glue', 'sst2', split='validation[:]')

    print("\nGenerating predictions for the evaluating data using the zero-shot text classifier...")
    zero_shot_eval_predictions = zero_shot_predict(zero_shot_classifier, zero_shot_labels, eval_data)

    print("\nZero-shot performance on training data...")
    display_result(zero_shot_eval_predictions)

if __name__ == "__main__":
    distil_bert_experiment()