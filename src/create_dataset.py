import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import os,transformers
from typing import List,Dict,Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

def convert_list_str_to_list_list_str(list_str):
  examples_list = []
  i = 0
  while i+4 <= len(list_str):
    examples_list.append(list_str[i:i+4])
    i += 5
  return examples_list


def entry_creator(example:List[str]):
  columns = ['source','morph_seg','target_gloss','translation']
  return {column:column_value for column,column_value in zip(columns,example)}


def entry_generator(examples:List[List[str]]):
  for example in examples:
    yield entry_creator(example)


def clean_example(example:Dict[str,str]): #This is used for dict_to_dataset's transform method!!!
  return {key:value[3:-1] for key,value in example.items()}


class Path_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: str, y=None):
        return {'main_path': X,
                'result_list': os.listdir(X),
                }


class List_To_Dicts(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):  # X should be a dictionary object with two items in it.
        main_path = X['main_path']
        file_names = X['result_list']
        result_dict = {}

        for file in file_names:
            with open(os.path.join(main_path, file), 'r') as f:
                file_lines = f.readlines()

            result_dict[file[:-4]] = convert_list_str_to_list_list_str(file_lines)

        return result_dict


class Dict_To_Dataset(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        datasets = {key: Dataset.from_generator(lambda: entry_generator(value)) for key, value in X.items()}
        datasets = {key: value.map(clean_example) for key, value in datasets.items()}
        return datasets


def main(file_path):
    data_processing_pipeline = Pipeline([
    ('path_transformer', Path_Transformer()),
    ('list_to_dicts', List_To_Dicts()),
    ('dict_to_dataset', Dict_To_Dataset()),
    ])
    return data_processing_pipeline.fit_transform(file_path)

