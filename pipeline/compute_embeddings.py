import sys
import pandas as pd
from diagnosis_clustering import load_umberto, add_bert_embeddings

text_col = 'testo_clean'
input_csv = sys.argv[1]
model_name = sys.argv[2]

output_csv = f"df_fse_embeddings.csv"

print(f"Loading original file: {input_csv}")
df = pd.read_csv(input_csv)

embedding_cols = [c for c in df.columns if c.startswith("U") and c[1:2].isdigit()]

if len(embedding_cols) == 0:
    print("No embedding columns found, computing BERT embeddings...")
    model, tokenizer = load_umberto(model_name)
    df = df[df[text_col].notnull()]
    df = add_bert_embeddings(model, tokenizer, df, text_col)
    print('Embeddings computed!')
    df.to_csv(output_csv, index=False)
    print(f"Embeddings saved to: {output_csv}")
else:
    print(f"Embeddings already present, nothing to do.")