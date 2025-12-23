import os
import sys
import pandas as pd

from pipeline.classification import run_full_experiment_resumable, setup_dataset

df = pd.read_csv('df_fse_second_level.csv')

weak_label_col = 'our_bronchio'
gold_label_col = 'bronchiolite'
text_col = 'testo_clean'
model = sys.argv[1]

dataset, df_bal = setup_dataset(df, text_col=text_col, gold_label_col=gold_label_col, weak_label_col=weak_label_col, clean=True)

all_results = run_full_experiment_resumable(df_bal, model, text_col, weak_label_col, gold_label_col,
                                  'classification_clean',
                                  dataset,
                                  weak_percentages=range(100, 110, 10),
                                  n_reps=1)

print('Finished!')