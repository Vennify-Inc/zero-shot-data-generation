"""
Explores using a zero-shot text classification model
to generate training data that a smaller supervised model can then use.

Uses a binary sentiment analysis dataset called Stanford Sentiment Treebank for demonstration

By Eric Fillion @ Vennify.ai
"""
from happytransformer import HappyTextClassification
from transformers import pipeline
from datasets import load_dataset
from textblob.classifiers import NaiveBayesClassifier

from util import *


def primary_experiment(number_of_train_cases):


    task = "zero-shot-classification"
    zero_shot_model = "facebook/bart-large-mnli"

    happy_model_type = "DISTILBERT"
    happy_model_name = "distilbert-base-uncased"

    happy_transformer_train_path = "happy_train.csv"
    textblob_train_path = "textblob_train.json"

    print("Loading a zero-shot transformer model...")
    zero_shot_classifier = pipeline(task, zero_shot_model)
    zero_shot_labels = ["negative", "positive"]

    print("Loading a supervised transformer model...")
    happy_tc = HappyTextClassification(happy_model_type, happy_model_name, num_labels=2)

    print("\nFetching data from Hugging Face's dataset distribution network...")
    train_split_string = "train[:" + str(number_of_train_cases) + "]"

    train_data = load_dataset('glue', 'sst2', split=train_split_string)
    eval_data = load_dataset('glue', 'sst2', split='validation[:]')

    print("\nGenerating predictions for the training data using the zero-shot text classifier...")
    zero_shot_train_predictions = zero_shot_predict(zero_shot_classifier, zero_shot_labels, train_data)

    print("\nZero-shot performance on training data...")
    display_result(zero_shot_train_predictions)

    print("\nGenerating predictions for the evaluating data using the zero-shot text classifier...")
    zero_shot_eval_predictions = zero_shot_predict(zero_shot_classifier, zero_shot_labels, eval_data)

    print("\nZero-shot performance on eval data...")
    display_result(zero_shot_eval_predictions)

    print("\nGenerating starting training data...")
    generate_training_csv(zero_shot_train_predictions, happy_transformer_train_path)

    generate_training_json(zero_shot_train_predictions, textblob_train_path)


    print("\n\n\n----------------TRAINING BERT----------------\n")

    happy_tc.train(happy_transformer_train_path)

    print("Generating predictions for training data...")
    bert_training_predictions = happy_tc_predict(happy_tc, train_data)
    display_result(bert_training_predictions)

    print("Generating predictions for eval data...", )
    bert_eval_predictions = happy_tc_predict(happy_tc, eval_data)
    display_result(bert_eval_predictions)

    print("\n\n\n----------------TRAINING NAIVE BAYES CLASSIFIER----------------\n")
    with open(textblob_train_path, 'r') as fp:
        naive_classifier = NaiveBayesClassifier(fp, format="json")

    print("Generating predictions for training data...")
    textblob_training_predictions = textblob_predict(naive_classifier, train_data)
    display_result(textblob_training_predictions)

    print("Generating predictions for eval data...")
    textblob_eval_predictions = textblob_predict(naive_classifier, eval_data)
    display_result(textblob_eval_predictions)

if __name__ == "__main__":
    primary_experiment(3)
