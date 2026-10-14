from pathlib import Path

import pandas as pd

from train import build_model_from_dataframe, load_dataset


def test_load_dataset_reads_cardio_csv():
    df = load_dataset(Path('CardioGoodFitness.csv'))
    assert not df.empty
    assert {'Product', 'Age', 'Gender', 'Education', 'MaritalStatus', 'Usage', 'Fitness', 'Income', 'Miles'} <= set(df.columns)


def test_build_model_from_dataframe_predicts_known_values():
    df = pd.read_csv('CardioGoodFitness.csv')
    artifact = build_model_from_dataframe(df)
    model = artifact['model']
    sample = df.drop(columns=['Product']).iloc[[0]].copy()
    prediction = model.predict(sample)[0]
    assert prediction in set(df['Product'])
