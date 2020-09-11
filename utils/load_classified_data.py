import pickle
import pandas as pd


def load_output_dataframe(path, trial_id):
    df = pd.read_csv(path)
    output = df[df['Trial_ID'] == trial_id]
    return output
