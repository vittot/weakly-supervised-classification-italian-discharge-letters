import torch
from utils import _get_device, sample_balanced_fse
from transformers import TrainingArguments, Trainer, TrainerCallback, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, make_scorer
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import os
import pickle
from sklearn.model_selection import KFold, StratifiedKFold
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import numpy as np
from evaluate import load
from sklearn.metrics import roc_curve
from torch.nn import functional as F
import pandas as pd
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from utils import clean_sentences

MODEL = 'Musixmatch/umberto-commoncrawl-cased-v1'
tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)

def get_umberto_pred(model, tokenizer, text):
  
  encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=512)
  input_ids = torch.tensor(encoded_input['input_ids']).unsqueeze(0).to(_get_device())
  att_mask = torch.tensor(encoded_input['attention_mask']).unsqueeze(0).to(_get_device())
  return model.forward(input_ids=input_ids, attention_mask=att_mask).logits[0][1].item()

def unfreeze_layers(model, prefix):
  for name, param in model.named_parameters():
      if name.startswith(prefix):
        param.requires_grad = True

class FreezingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model, **kwargs):
      if state.epoch==2:
        unfreeze_layers(model, 'roberta.encoder.layer.11')
        print('unfreezing layer 11')
      elif state.epoch==4:
        unfreeze_layers(model, 'roberta.encoder.layer.10')
        print('unfreezing layer 10')

metrics = ["accuracy", "recall", "precision", "f1", "roc_auc"]
metric = {met: load(met) for met in metrics}

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to probabilities
    logits = torch.from_numpy(logits)
    probabilities_scores = F.softmax(logits, dim=-1).numpy()
    proba = probabilities_scores[:, 1]  # probability of class 1

    # Find optimal threshold using Youden's J statistic
    fpr, tpr, threshold = roc_curve(labels, proba)
    j_scores = tpr - fpr
    cutoff = sorted(zip(j_scores, threshold))[-1][1]
    predictions = (proba > cutoff).astype(int)

    # Compute metrics
    metric_res = {}
    for met in metrics[:-1]:  # skip roc_auc for now
        try:
            metric_res[met] = metric[met].compute(predictions=predictions, references=labels)[met]
        except Exception as e:
            print(f"{met} failed: {e}")

    # Compute ROC AUC separately
    try:
        metric_res["roc_auc"] = metric["roc_auc"].compute(prediction_scores=proba, references=labels)["roc_auc"]
    except Exception as e:
        print(f"roc_auc failed: {e}")
        metric_res["roc_auc"] = "ERROR"

    return metric_res

def setup_dataset(df, text_col, gold_label_col, weak_label_col):
    if text_col == 'testo_clean':
        df['testo_clean'] = df['testo'].apply(lambda x: clean_sentences(x,False))
    elif text_col == 'testo_clean_wod':
         df['testo_clean_wod'] = df['testo'].apply(lambda x: clean_sentences(x,True))

    df_bal = sample_balanced_fse(df, text_col, gold_label_col, weak_label_col)
    dataset = Dataset.from_pandas(df_bal, preserve_index=False)

    def transform_labels(row):
        num = 1 if row[weak_label_col] else 0
        return {'labels': num}

    def tokenize_data(row):
        return tokenizer(row[text_col], padding='max_length', truncation=True, max_length=512)

    dataset = dataset.map(tokenize_data, batched=True)

    df_bal[gold_label_col] = df_bal[gold_label_col].astype('bool')
    df_bal[weak_label_col] = df_bal[weak_label_col].astype('bool')
    return dataset, df_bal

def make_mixed_label_column(df_bal, dataset, weak_col, gold_col,
                            weak_pct, random_state=1234):
    """
    Create a new column in df_bal and dataset with a given percentage of weak labels.
    
    weak_pct: percentage of weak labels in [0, 100].
    weak_col: name of weak label column
    gold_col: name of gold label column
    """
    rng = np.random.RandomState(random_state)
    n = len(df_bal)

    # Name of the new training-label column
    mixed_col = "mixed_label"

    # Start from gold labels everywhere
    df_bal[mixed_col] = df_bal[gold_col].astype(int)

    # Number of samples that will use weak labels
    n_weak = int(round(weak_pct / 100.0 * n))
    if n_weak > 0:
        weak_idx = rng.choice(n, size=n_weak, replace=False)
        df_bal.loc[df_bal.index[weak_idx], mixed_col] = df_bal.loc[df_bal.index[weak_idx], weak_col].astype(int)

    # Mirror this column into the HF dataset, assuming same ordering & length
    def add_mixed_label(example, idx):
        example[mixed_col] = int(df_bal.iloc[idx][mixed_col])
        return example

    dataset_with_mixed = dataset.map(add_mixed_label, with_indices=True)

    return df_bal, dataset_with_mixed, mixed_col

def make_transform_labels(label_col):
    def transform_labels(row):
        label = row[label_col]
        num = 1 if label else 0
        return {'labels': num}
    return transform_labels

def run_cv(df_bal, model, weak_label_col, gold_label_col, text_col,
           experiment_name, dataset, sfreeze=True):

  df_bal[weak_label_col] = df_bal[weak_label_col].astype(int)
  transform_labels_mixed = make_transform_labels(weak_label_col)

  folds = StratifiedKFold(n_splits=10, random_state=1234, shuffle=True)
  crs = []       # classification reports vs mixed label
  aucs = []      # AUCs vs mixed label
  crs_or = []    # classification reports vs gold
  aucs_or = []   # AUCs vs gold

  unfreeze_call = FreezingCallback()

  splits = folds.split(
      np.zeros(len(df_bal[text_col].reset_index(drop=True))),
      df_bal[weak_label_col].reset_index(drop=True)
  )

  for (i, (train_idxs, test_idxs)) in tqdm(enumerate(splits), total=10):

    train_dataset = dataset.select(train_idxs)
    test_dataset  = dataset.select(test_idxs)

    remove_columns = [weak_label_col, text_col, gold_label_col]

    train_dataset = train_dataset.map(transform_labels_mixed,
                                      remove_columns=remove_columns)
    val_dataset   = test_dataset.map(transform_labels_mixed,
                                     remove_columns=remove_columns)

    args = TrainingArguments(
      "bert_" + experiment_name,
      eval_strategy="epoch",
      do_eval=True,
      logging_steps=50,
      learning_rate=1e-3,
      num_train_epochs=6,
      weight_decay=0.01,
      push_to_hub=False,
      per_device_train_batch_size=8,
      save_strategy="no",
      metric_for_best_model='roc_auc',
      report_to=[]
    )

    umberto = AutoModelForSequenceClassification.from_pretrained(
      model,
      num_labels=2,
      output_attentions=False,
      output_hidden_states=False,
    )
    print("Initial device:", next(umberto.parameters()).device)

    for name, param in umberto.named_parameters():
      if not name.startswith("classifier"):
          param.requires_grad = False
      else:
        param.requires_grad = True

    callbacks = [unfreeze_call] if sfreeze else None

    trainer = Trainer(
        model=umberto,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()
    umberto.eval()
    umberto.to(_get_device())

    # --- probs on original (unmapped) test_dataset ---
    y_proba = [get_umberto_pred(umberto, tokenizer, x) for x in test_dataset[text_col]]

    # ---------- metrics vs MIXED label ----------
    y_test_mixed = [1 if y else 0 for y in test_dataset[weak_label_col]]
    fpr, tpr, threshold = roc_curve(y_test_mixed, y_proba)
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, threshold))
    cutoff = j_ordered[-1][1]
    y_pred_mixed = [1 if p > cutoff else 0 for p in y_proba]

    aucs.append(roc_auc_score(y_test_mixed, y_proba))
    crs.append(classification_report(test_dataset[weak_label_col],
                                     y_pred_mixed,
                                     output_dict=True))

    # ---------- metrics vs GOLD label ----------
    y_test_gold = [1 if y else 0 for y in test_dataset[gold_label_col]]
    fpr, tpr, threshold = roc_curve(y_test_gold, y_proba)
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, threshold))
    cutoff = j_ordered[-1][1]
    y_pred_gold = [1 if p > cutoff else 0 for p in y_proba]

    aucs_or.append(roc_auc_score(y_test_gold, y_proba))
    crs_or.append(classification_report(test_dataset[gold_label_col],
                                        y_pred_gold,
                                        output_dict=True))

  return {
      "aucs_mixed": aucs,
      "crs_mixed": crs,
      "aucs_gold": aucs_or,
      "crs_gold": crs_or,
  }

def run_full_experiment(model,
                        text_col,
                        base_experiment_name,
                        dataset,
                        df_bal,
                        weak_label_col,
                        gold_label_col,
                        weak_percentages=range(0, 110, 10),
                        n_reps=5):

    """
    Runs the full experiment:
      - for each weak_pct in weak_percentages
      - for each rep in 0..n_reps-1
        * create a new mixed-label column with a different random_state
        * run 10-fold CV
        * store per-fold metrics (mixed + gold)
    Saves everything into results_<base_experiment_name>.pkl
    """

    all_results = {}  # structure: {weak_pct: {rep: cv_result_dict}}

    for weak_pct in tqdm(weak_percentages):
        all_results[weak_pct] = {}

        for rep in tqdm(range(n_reps)):
            seed = 1234 + rep

            # 1) create a specific mixed column for this (weak_pct, rep)
            df_bal, dataset_mixed, mixed_col = make_mixed_label_column(
                df_bal=df_bal,
                dataset=dataset,
                weak_col=weak_label_col,
                gold_col=gold_label_col,
                weak_pct=weak_pct,
                random_state=seed
            )

            # 2) build experiment name
            exp_name = f"{base_experiment_name}_weak{weak_pct}_rep{rep}"

            # 3) returns per-fold metrics
            cv_results = run_cv(
                df_bal,
                model=model,
                weak_label_col=mixed_col,          
                gold_label_col=gold_label_col,
                text_col=text_col,
                experiment_name=base_experiment_name,
                dataset=dataset_mixed,
                sfreeze=True
            )
            all_results[weak_pct][rep] = cv_results

            print(f"Done: weak={weak_pct}%, rep={rep}")

    # 4) save in one file
    out_fname = f"results_{base_experiment_name}.pkl"
    with open(out_fname, "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nSaved all results to {out_fname}")
    return all_results

def run_full_experiment_resumable(df_bal, model, text_col, weak_label_col, gold_label_col,
                                  base_experiment_name,
                                  dataset,
                                  weak_percentages=range(0, 110, 10),
                                  n_reps=5):

    results_path = f"results_{base_experiment_name}.pkl"

    # 1) Load previous results if they exist
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            all_results = pickle.load(f)
        print(f"Loaded existing results from {results_path}")
    else:
        all_results = {}

    for weak_pct in tqdm(weak_percentages):
        if weak_pct not in all_results:
            all_results[weak_pct] = {}

        for rep in tqdm(range(n_reps)):
            if rep in all_results[weak_pct]:
                print(f"Skipping weak={weak_pct}%, rep={rep} (already done)")
                continue

            seed = 1234 + rep

            # 2) Create the mixed label column for this combo
            df_bal, dataset_mixed, mixed_col = make_mixed_label_column(
                df_bal=df_bal,
                dataset=dataset,
                weak_col=weak_label_col,
                gold_col=gold_label_col,
                weak_pct=weak_pct,
                random_state=seed
            )

            exp_name = f"{base_experiment_name}_weak{weak_pct}_rep{rep}"

            # 3) Run CV -> get per-fold metrics dict
            cv_results = run_cv(
                df_bal,
                model=model,
                weak_label_col=mixed_col,          
                gold_label_col=gold_label_col,
                text_col=text_col,
                experiment_name=base_experiment_name,
                dataset=dataset_mixed,
                sfreeze=True
            )

            # 4) Store in memory
            all_results[weak_pct][rep] = cv_results

            # 5) Save after each combo
            with open(results_path, "wb") as f:
                pickle.dump(all_results, f)

            print(f"Done and saved weak={weak_pct}%, rep={rep}")

    print(f"\nFinal results saved to {results_path}")
    return all_results

def run_experiment_all_weak_sources_100(
        df,
        model,
        text_col,
        weak_label_cols,      
        gold_label_col,
        base_experiment_name,
        n_reps=5
):
    """
    For each weak label column in weak_label_cols, run CV using 100% weak labels 
    (no mixing with gold labels) and evaluate on gold_label_col.

    Results are stored as:
        all_results[weak_col][rep] = cv_results
    and saved/resumed from a pickle.
    """

    results_path = f"results_{base_experiment_name}_allweak.pkl"

    # 1) Load previous results if they exist
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            all_results = pickle.load(f)
        print(f"Loaded existing results from {results_path}")
    else:
        all_results = {}

    # 2) Loop over all weak-label sources (clustering settings)
    for weak_col in tqdm(weak_label_cols, desc="Weak label sources"):

        if weak_col not in all_results:
            all_results[weak_col] = {}

        dataset, df_bal = setup_dataset(df, text_col=text_col, gold_label_col=gold_label_col, weak_label_col=weak_col)

        for rep in tqdm(range(n_reps), desc=f"{weak_col}", leave=False):

            if rep in all_results[weak_col]:
                print(f"Skipping {weak_col}, rep={rep} (already done)")
                continue

            seed = 1234 + rep

            # 100% weak labels
            df_bal_tmp, dataset_weak, mixed_col = make_mixed_label_column(
                df_bal=df_bal,
                dataset=dataset,
                weak_col=weak_col,
                gold_col=gold_label_col,
                weak_pct=100,        
                random_state=seed
            )

            exp_name = f"{base_experiment_name}_{weak_col}_weak100_rep{rep}"

            cv_results = run_cv(
                df_bal_tmp,
                model=model,
                weak_label_col=mixed_col,      
                gold_label_col=gold_label_col,
                text_col=text_col,
                experiment_name=exp_name,
                dataset=dataset_weak,
                sfreeze=True
            )

            all_results[weak_col][rep] = cv_results

            # save after each combo
            with open(results_path, "wb") as f:
                pickle.dump(all_results, f)

            print(f"Done and saved {weak_col}, rep={rep}")

    print(f"\nFinal results saved to {results_path}")
    return all_results


def run_stratified_cv_pediatria(
    *,
    dataset, 
    df_bal,  
    label_col_weak: str,          
    label_col_gold: str,          
    model_path: str,
    text_col: str = "testo_clean",
    pediatria_col: str = "pediatria",
    n_splits: int = 10,
    random_state: int = 1234,
    output_dir: str = "bert_cv_ped",
    training_args_kwargs: Optional[Dict[str, Any]] = None,
    remove_columns: Optional[List[str]] = None,
    transform_labels: Optional[Callable[..., Any]] = None,
    compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
    callbacks: Optional[List[Any]] = None,
    freeze_except_classifier: bool = True,
    save_every_split: bool = True,
    save_prefix: str = "cv_ped",
    save_dir: str = ".",
    verbose: bool = True,
) -> Dict[str, Dict[str, List[Any]]]:
    """
    CV test for Pediatrics vs Non-Pediatrics ER/subsets.

    Returns a dict:
      {
        "ped_weak":    {"aucs": [...], "crs": [...]},
        "ped_gold":    {"aucs": [...], "crs": [...]},
        "nonped_weak": {"aucs": [...], "crs": [...]},
        "nonped_gold": {"aucs": [...], "crs": [...]},
      }

    """


    if training_args_kwargs is None:
        training_args_kwargs = {}

    if remove_columns is None:
        remove_columns = ["our_bronchio", "bronchiolite", "testo_clean", "localita", "azienda", "pediatria"]

    os.makedirs(save_dir, exist_ok=True)

    results: Dict[str, Dict[str, List[Any]]] = {
        "ped_weak": {"aucs": [], "crs": []},
        "ped_gold": {"aucs": [], "crs": []},
        "nonped_weak": {"aucs": [], "crs": []},
        "nonped_gold": {"aucs": [], "crs": []},
    }

    def _youden_cutoff(y_true: Sequence[int], y_score: Sequence[float]) -> float:
        fpr, tpr, thr = roc_curve(y_true, y_score)
        return float(thr[np.argmax(tpr - fpr)])

    def _eval_with_youden(y_true: Sequence[int], y_score: Sequence[float]) -> Tuple[float, Dict[str, Any]]:
        cutoff = _youden_cutoff(y_true, y_score)
        y_pred = [1 if p > cutoff else 0 for p in y_score]
        auc = float(roc_auc_score(y_true, y_pred, average="macro"))
        cr = classification_report(y_true, y_pred, output_dict=True)
        return auc, cr

    def _dump(obj: Any, filename: str) -> None:
        path = os.path.join(save_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def _eval_subset(
        subset,
        y_col: str,
        bucket_key: str,
        split_idx: int,
        label_name: str,
        model,
        tokenizer
    ) -> None:
        texts = subset[text_col]
        y_score = [float(get_umberto_pred(model, tokenizer, t)) for t in texts]
        y_true = [1 if y else 0 for y in subset[y_col]]

        auc, cr = _eval_with_youden(y_true, y_score)
        results[bucket_key]["aucs"].append(auc)
        results[bucket_key]["crs"].append(cr)

        if verbose:
            aucs = np.array(results[bucket_key]["aucs"], dtype=float)
            print(f"[split {split_idx}] {label_name} | mean AUC={aucs.mean():.4f} std={aucs.std():.4f}")

    y_for_split = df_bal[label_col_weak].reset_index(drop=True)
    folds = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    for split_idx, (train_idxs, test_idxs) in enumerate(folds.split(np.zeros(len(y_for_split)), y_for_split)):
        train_dataset = dataset.select(train_idxs)
        test_dataset = dataset.select(test_idxs)

        test_ped = test_dataset.filter(lambda x: bool(x[pediatria_col]))
        test_nonped = test_dataset.filter(lambda x: not bool(x[pediatria_col]))

        train_dataset = train_dataset.map(transform_labels, remove_columns=remove_columns)
        val_dataset = test_dataset.map(transform_labels, remove_columns=remove_columns)

        args = TrainingArguments(
            output_dir,
            evaluation_strategy="epoch",
            do_eval=True,
            logging_steps=50,
            learning_rate=1e-3,
            num_train_epochs=6,
            weight_decay=0.01,
            push_to_hub=False,
            per_device_train_batch_size=8,
            save_strategy="no",
            metric_for_best_model="accuracy",
            **training_args_kwargs,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        if freeze_except_classifier:
            for name, param in model.named_parameters():
                param.requires_grad = bool(name.startswith("classifier"))

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        trainer.train()

        _eval_subset(test_ped, label_col_weak, "ped_weak", split_idx, "PED - weak", model, tokenizer)
        _eval_subset(test_ped, label_col_gold, "ped_gold", split_idx, "PED - gold", model, tokenizer)

        _eval_subset(test_nonped, label_col_weak, "nonped_weak", split_idx, "NONPED - weak", model, tokenizer)
        _eval_subset(test_nonped, label_col_gold, "nonped_gold", split_idx, "NONPED - gold", model, tokenizer)
        if save_every_split:
            _dump(results["nonped_weak"]["crs"],  f"crs_{save_prefix}.bin")
            _dump(results["nonped_weak"]["aucs"], f"aucs_{save_prefix}.bin")

            _dump(results["nonped_gold"]["crs"],  f"crs_or{save_prefix}.bin")
            _dump(results["nonped_gold"]["aucs"], f"aucs_or{save_prefix}.bin")

            _dump(results["ped_weak"]["crs"],  f"crs_ped_{save_prefix}.bin")
            _dump(results["ped_weak"]["aucs"], f"aucs_ped_{save_prefix}.bin")

            _dump(results["ped_gold"]["crs"],  f"crs_ped_or{save_prefix}.bin")
            _dump(results["ped_gold"]["aucs"], f"aucs_ped_or{save_prefix}.bin")

    return results


def leave_one_group_out_cv(
    *,
    dataset,                          
    groups: Sequence[str],            
    group_col: str,                   
    model_path: str,                  
    text_col: str,                    
    label_col_weak: str,              
    label_col_gold: str,              
    transform_labels: Callable[..., Any],
    compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
    callbacks: Optional[List[Any]] = None,
    remove_columns: Optional[List[str]] = None,
    freeze_except_classifier: bool = True,
    training_args_kwargs: Optional[Dict[str, Any]] = None,
    output_dir_prefix: str = "bert_cv_",
    save_model_each_group: bool = True,
    save_metrics_each_group: bool = True,
    experiment_name: str = "",
    save_dir: str = ".",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Leave-one-group-out CV for a categorical grouping column (e.g., LHU or hospital).

    Returns:
      {
        "<group>": {
          "weak": {"auc": float|None, "cr": dict},
          "gold": {"auc": float|None, "cr": dict},
          "n_train": int,
          "n_test": int,
          "pos_train": int|None,
          "pos_test_weak": int,
          "pos_test_gold": int,
        },
        ...
      }

    """
    if training_args_kwargs is None:
        training_args_kwargs = {}
    if remove_columns is None:
        remove_columns = ["our_bronchio", "bronchiolite", "testo_clean", "localita", "azienda"]

    os.makedirs(save_dir, exist_ok=True)

    def youden_cutoff(y_true: List[int], y_score: List[float]) -> float:
        fpr, tpr, thr = roc_curve(y_true, y_score)
        return float(thr[np.argmax(tpr - fpr)])

    def eval_labelset(
        y_true: List[int],
        y_proba: List[float],
        y_true_for_report,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        cutoff = youden_cutoff(y_true, y_proba)
        y_pred = [1 if p > cutoff else 0 for p in y_proba]
        auc = float(roc_auc_score(y_true, y_proba, average="macro")) if sum(y_true) > 0 else None
        cr = classification_report(y_true_for_report, y_pred, output_dict=True)
        return auc, cr

    results: Dict[str, Dict[str, Any]] = {}

    for i, g in enumerate(groups):
        train_dataset = dataset.filter(lambda x: x[group_col] != g)
        test_dataset = dataset.filter(lambda x: x[group_col] == g)

        if verbose:
            print(f"[{i+1}/{len(groups)}] {group_col}={g} | train={len(train_dataset)} test={len(test_dataset)}")

        train_ds = train_dataset.map(transform_labels, remove_columns=remove_columns)
        val_ds = test_dataset.map(transform_labels, remove_columns=remove_columns)

        args = TrainingArguments(
            output_dir=f"{output_dir_prefix}{group_col}_{g}",
            evaluation_strategy="epoch",
            do_eval=True,
            logging_steps=50,
            learning_rate=1e-3,
            num_train_epochs=6,
            weight_decay=0.01,
            push_to_hub=False,
            per_device_train_batch_size=8,
            save_strategy="no",
            metric_for_best_model="accuracy",
            **training_args_kwargs,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        if freeze_except_classifier:
            for name, param in model.named_parameters():
                param.requires_grad = bool(name.startswith("classifier"))

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        trainer.train()

        if save_model_each_group:
            trainer.save_model(f"{output_dir_prefix}{group_col}_{g}")

        y_proba = [float(get_umberto_pred(model, tokenizer, x)) for x in test_dataset[text_col]]

        y_true_weak = [1 if y else 0 for y in test_dataset[label_col_weak]]
        auc_weak, cr_weak = eval_labelset(y_true_weak, y_proba, test_dataset[label_col_weak])

        y_true_gold = [1 if y else 0 for y in test_dataset[label_col_gold]]
        auc_gold, cr_gold = eval_labelset(y_true_gold, y_proba, test_dataset[label_col_gold])

        results[g] = {
            "weak": {"auc": auc_weak, "cr": cr_weak},
            "gold": {"auc": auc_gold, "cr": cr_gold},
            "n_train": int(len(train_dataset)),
            "n_test": int(len(test_dataset)),
            "pos_train": int(sum(train_ds["labels"])) if "labels" in train_ds.column_names else None,
            "pos_test_weak": int(sum(y_true_weak)),
            "pos_test_gold": int(sum(y_true_gold)),
        }

        if save_metrics_each_group:
            with open(os.path.join(save_dir, f"crs_{experiment_name}{g}.bin"), "wb") as fp:
                pickle.dump(cr_weak, fp)
            with open(os.path.join(save_dir, f"aucs_{experiment_name}{g}.bin"), "wb") as fp:
                pickle.dump(auc_weak, fp)

            with open(os.path.join(save_dir, f"crs_or{experiment_name}{g}.bin"), "wb") as fp:
                pickle.dump(cr_gold, fp)
            with open(os.path.join(save_dir, f"aucs_or{experiment_name}{g}.bin"), "wb") as fp:
                pickle.dump(auc_gold, fp)

    return results