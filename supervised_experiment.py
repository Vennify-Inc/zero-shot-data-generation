"""
Determines how well the models perform using data that's correctly labeled
By Eric Fillion @ Vennify.ai
"""
from happytransformer import HappyTextClassification
from datasets import load_dataset
from textblob.classifiers import NaiveBayesClassifier

from util import *


def generate_labeled_csv(train_data, happy_transformer_train_path):

    with open(happy_transformer_train_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["text", "label"])
        for case in train_data:
            writter.writerow([case["sentence"], case["label"]])


def generate_labeled_json(train_data, textblob_train_path):
    textblob_data = []

    for case in train_data:
        textblob_data.append({
            'text': case["sentence"],
            'label': "positive" if case["label"] else "negative"
        })
    with open(textblob_train_path, 'w') as f_out:
        json.dump(textblob_data, f_out)

def supervised(number_of_train_cases):
    happy_model_type = "DISTILBERT"
    happy_model_name = "distilbert-base-uncased"

    happy_transformer_train_path = "happy_train.csv"
    textblob_train_path = "textblob_train.json"

    print("Loading a supervised transformer model...")
    happy_tc = HappyTextClassification(happy_model_type, happy_model_name, num_labels=2)

    print("\nFetching data from Hugging Face's dataset distribution network...")

    train_split_string = "train[:" + str(number_of_train_cases) + "]"
    train_data = load_dataset('glue', 'sst2', split=train_split_string)
    eval_data = load_dataset('glue', 'sst2', split='validation[:]')

    print("\nGenerating starting training data...")
    generate_labeled_csv(train_data, happy_transformer_train_path)

    generate_labeled_json(train_data, textblob_train_path)


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
    supervised(3)
