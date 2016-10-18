import tweepy
import yaml
import csv


#TODO: function for extracting training data from yamls
#TODO: function for extracting training data from .csv (need to know formatting)
#TODO: tweet to predictor parser

#def prepare_test_data():
#    """test function to prepare movie review data.
#    Uses dataset from http://ai.stanford.edu/~amaas/data/sentiment/
#    for testing until tweet dataset is large enough"""
    

def signal_from_yaml(yaml_file_path):
    all_yamls = yaml.load(open(yaml_file_path, "r"))
    return [x for x in all_yamls.values() if x.retweeted]

def bkg_from_yaml(yaml_file_path):
    all_yamls = yaml.load(open(yaml_file_path, "r"))
    return [x for x in all_yamls.values() if not x.retweeted]

def signal_and_bkg_to_csv(signal, bkg, csv_file_path="tweepy_data.csv"):
    signal_for_file = [{"text": x.text, "classification": 1} for x in signal]
    bkg_for_file = [{"text": x.text, "classification": 0} for x in bkg]
    list_for_file = []
    list_for_file.extend(signal_for_file)
    list_for_file.extend(bkg_for_file)
    fieldnames = list_for_file[0].keys()
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in list_for_file:
            writer.writerow(row)

