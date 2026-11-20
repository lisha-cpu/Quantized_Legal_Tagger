from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading LexGLUE / LEDGAR...")
    ds = load_dataset("lex_glue", "ledgar")
    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()
    # LexGLUE LEDGAR typically provides text and label fields.
    print("Columns:", train.columns.tolist())
    text_col = "text"
    label_col = "label"
    if text_col not in train or label_col not in train:
        raise ValueError(f"Unexpected columns: {train.columns.tolist()}")
    train = train[[text_col,label_col]].dropna()
    test = test[[text_col,label_col]].dropna()
    # Keep manageable subset and restrict to the most frequent labels for a laptop run.
    counts = train[label_col].value_counts()
    chosen = counts.head(8).index.tolist()
    train = train[train[label_col].isin(chosen)]
    test = test[test[label_col].isin(chosen)]
    train = train.sample(min(2500, len(train)), random_state=42)
    test = test.sample(min(500, len(test)), random_state=42)
    # Convert numeric label IDs to readable names when available.
    names = ds["train"].features[label_col].names
    train["label_name"] = train[label_col].map(lambda x: names[int(x)])
    test["label_name"] = test[label_col].map(lambda x: names[int(x)])
    train.to_csv(OUT/"train.csv", index=False)
    test.to_csv(OUT/"test.csv", index=False)
    pd.Series(chosen).to_csv(OUT/"selected_label_ids.csv",index=False)
    print(f"Saved {len(train)} training and {len(test)} test rows; labels:", train.label_name.unique().tolist())
if __name__=="__main__":
    main()
