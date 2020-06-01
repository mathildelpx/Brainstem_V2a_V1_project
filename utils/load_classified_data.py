import pickle
import pandas as pd


def load_output_dataframe(path, trial):
    with open(path, 'rb') as f:
        df = pickle.load(f)
    output = df[df['Trial_ID'] == trial]
    return output

