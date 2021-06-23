"""
Contains helper functions
"""
import csv
import json
from tqdm import tqdm


def zero_shot_predict(zero_shot_classifier, zero_shot_labels, data):
    """
    :param zero_shot_classifier: a pipleine for a zero-shot classification created using Hugging Face's transformers library
    :param zero_shot_labels: a list of strings for the zero-shot labels
    :param data: a dataset creating using Hugging Face's datasets library
    :return: a list of dictionaries with the following keys: sentence, prediction and answer
    """
    labeled_cases = []
    for case in tqdm(data):
        prediction = zero_shot_classifier(case["sentence"], zero_shot_labels)
        prediction_string = prediction["labels"][0]

        # get the index number of the predicted label
        prediction_int = 0
        if prediction_string == zero_shot_labels[1]:
            prediction_int = 1

        result = {'sentence': case["sentence"], 'prediction': prediction_int, "answer": case["label"]}
        labeled_cases.append(result)

    return labeled_cases


def display_result(predictions):
    """
    Displays the performance of a set of predictions.
    :param predictions: a list of dictionaries with the following keys: sentence, prediction and answer
    :return:  None
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for case in predictions:
        if case['prediction'] == case['answer']:
            #  Prediction is correct
            if case['prediction'] == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            #  Prediction is wrong
            if case['prediction'] == 1:
                false_positive += 1
            else:
                false_negative += 1

    total = true_positive + true_negative + false_positive + false_negative
    percentage = (true_positive + true_negative) / total
    print("Percentage correct: ", str(percentage*100) + "%")
    print("true_positive ", true_positive)
    print("true_negative ", true_negative)
    print("false_positive ", false_positive)
    print("false_negative ", false_negative)


def generate_training_csv(predictions, path):
    """
    Creates a CSV file for in a format that's understandable by HappyTextClassification objects for training
    :param path: a string that contains the path to a CSV file.
    :param predictions: a list of dictionaries with the two keys: sentence and prediction
    :return:
    """
    with open(path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["text", "label"])
        for case in predictions:
            writter.writerow([case["sentence"], case["prediction"]])

def generate_training_json(predictions, path):
    """
    Creates a CSV file for in a format that's understandable by HappyTextClassification objects for training
    :param path: a string that contains the path to a JSON file.
    :param predictions: a list of dictionaries with the keys sentence and prediction
    :return:
    """
    textblob_data = []

    for case in predictions:
        textblob_data.append({
            'text': case["sentence"],
            'label': "positive" if case["prediction"] else "negative"
        })
    with open(path, 'w') as f_out:
        json.dump(textblob_data, f_out)


def happy_tc_predict(happy_tc, data):
    """
    :param happy_tc: a HappyTextClassification object
    :param data: a Hugging Face dataset
    :return: a list of dictionaries with the following keys: sentence, prediction and answer
    """
    predictions = []
    for case in tqdm(data):
        result = happy_tc.classify_text(case["sentence"])
        prediction = result.label
        prediction_int = 0
        if prediction == "LABEL_1":
            prediction_int = 1

        result = {'sentence': case["sentence"], 'prediction': prediction_int, "answer": case["label"]}
        predictions.append(result)

    return predictions


def textblob_predict(textblob, data):
    """

    :param textblob: A TextBlob NaiveBayesClassifier
    :param data: a Hugging Face dataset
    :return: a list of dictionaries with the following keys: sentence, prediction and answer
    """
    predictions = []
    for case in tqdm(data):
        prediction = textblob.classify(case["sentence"])

        prediction_int = 0
        if prediction == "positive":
            prediction_int = 1

        result = {'sentence': case["sentence"], 'prediction': prediction_int, "answer": case["label"]}
        predictions.append(result)
    return predictions
