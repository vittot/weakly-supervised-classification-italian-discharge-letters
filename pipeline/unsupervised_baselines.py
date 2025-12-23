from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, StratifiedKFold
import re
import numpy as np
import pandas as pd
import pickle
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.utils import _get_device, sample_balanced_fse
import math
from datasets import Dataset
from typing import List, Dict

LLM_ID = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
SYSTEM_MSG = (
    "Sei un assistente clinico. "
    "Devi decidere se il seguente testo indica la PRESENZA della condizione descritta. "
    "Rispondi solo con 'Sì' o 'No'."
)

def load_llm():
    device = _get_device()
    quant = None
    try:
        if device == "cuda":
            import bitsandbytes as bnb  
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16)
    except Exception:
        quant = None
    tok = AutoTokenizer.from_pretrained(LLM_ID, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if quant:
        model = AutoModelForCausalLM.from_pretrained(LLM_ID, quantization_config=quant, device_map="auto")
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=dtype, device_map="auto" if device=="cuda" else None)
        if device == "cpu": model.to("cpu")
    model.eval()
    return model, tok, device

def chat_prompt(text, label_name, tok):
    msgs = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",
         "content": f"Condizione: {label_name}\n\nTesto:\n\"\"\"\n{text}\n\"\"\"\n\nRispondi solo con 'Sì' o 'No'."}
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def option_logprob(model, tok, prompt, opt, device):
    opt = " " + opt.strip()
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    f_ids = tok(prompt + opt, return_tensors="pt", add_special_tokens=False).to(device)
    opt_len = f_ids.input_ids.shape[1] - p_ids.input_ids.shape[1]
    if opt_len <= 0: return -1e9
    out = model(f_ids.input_ids)
    lp  = torch.log_softmax(out.logits[:, :-1, :], dim=-1)  # [1, L-1, V]
    labels = f_ids.input_ids[:, -opt_len:]
    # positions corresponding to option tokens
    start = lp.shape[1] - opt_len
    s = 0.0
    for i in range(opt_len):
        s += lp[0, start + i, labels[0, i]].item()
    return float(s)

@torch.no_grad()
def proba_si(model, tok, text, label_name, device):
    prompt = chat_prompt(text, label_name, tok)
    l_si = option_logprob(model, tok, prompt, "Sì", device)
    l_no = option_logprob(model, tok, prompt, "No", device)
    m = max(l_si, l_no)
    return math.exp(l_si - m) / (math.exp(l_si - m) + math.exp(l_no - m))

def run_cv_zero_shot_llm(df_fse, MALATTIA, GOLD_MALATTIA, TEXT_COL, experiment_name, n_splits=10, seed=1234):
    df_bal = sample_balanced_fse(df_fse, TEXT_COL, MALATTIA, GOLD_MALATTIA)
    dataset = Dataset.from_pandas(df_bal, preserve_index=False)
    # Ensure binary ints
    df_bal[MALATTIA] = df_bal[MALATTIA].astype(int)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(folds.split(np.zeros(len(df_bal)), df_bal[MALATTIA].to_numpy()))

    model, tok, device = load_llm()

    aucs, crs = [], []
    aucs_or, crs_or = [], []

    for i, (tr_idx, te_idx) in tqdm(enumerate(splits), total=len(splits)):
        train_ds = dataset.select(tr_idx)
        test_ds  = dataset.select(te_idx)

        # ---- score train & choose threshold (NO peeking at test labels)
        y_train = [1 if y else 0 for y in train_ds[MALATTIA]]
        p_train = [proba_si(model, tok, x, MALATTIA, device) for x in train_ds[TEXT_COL]]
        fpr, tpr, thr = roc_curve(y_train, p_train)
        j = tpr - fpr
        cutoff = thr[np.argmax(j)]  # Youden J on train only

        # ---- score test fold
        y_test = [1 if y else 0 for y in test_ds[MALATTIA]]
        p_test = [proba_si(model, tok, x, MALATTIA, device) for x in test_ds[TEXT_COL]]
        y_pred = [1 if p > cutoff else 0 for p in p_test]
        aucs.append(roc_auc_score(y_test, p_test))
        crs.append(classification_report(y_test, y_pred, output_dict=True))

        # ---- original (gold) labels, same cutoff learned on TRAIN(MALATTIA)
        y_gold = [1 if y else 0 for y in test_ds[GOLD_MALATTIA]]
        aucs_or.append(roc_auc_score(y_gold, p_test))
        y_pred_gold = [1 if p > cutoff else 0 for p in p_test]
        crs_or.append(classification_report(y_gold, y_pred_gold, output_dict=True))

        # persist running
        with open(f'crs_{experiment_name}.bin', 'wb') as fp:   pickle.dump(crs, fp)
        with open(f'aucs_{experiment_name}.bin', 'wb') as fp:  pickle.dump(aucs, fp)
        with open(f'crs_or{experiment_name}.bin', 'wb') as fp: pickle.dump(crs_or, fp)
        with open(f'aucs_or{experiment_name}.bin', 'wb') as fp:pickle.dump(aucs_or, fp)

    return {
        "AUC mean ± sd": (float(np.mean(aucs)), float(np.std(aucs))),
        "AUC_OR mean ± sd": (float(np.mean(aucs_or)), float(np.std(aucs_or))),
    }




def ensure_rule_col(
    df: pd.DataFrame,
    rule_col: str,
    source_col: str,
    rules: List[Dict[str, List[str]]],
):
    """
    Build a boolean rule column based on multiple rules.

    Each rule is a dict with:
      - 'pos': list of positive keywords (all must be present)
      - 'neg': list of negative keywords (none must be present; optional)

    The rule applies if at least one rule matches.
    """

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).lower()

    def rule_applies(text: str) -> bool:
        for rule in rules:
            pos = rule.get("pos", [])
            neg = rule.get("neg", [])

            # all positive keywords must be present
            if not all(k in text for k in pos):
                continue

            # no negative keyword must be present
            if any(k in text for k in neg):
                continue

            return True  # at least one rule applies

        return False

    normalized_text = df[source_col].apply(normalize)
    df[rule_col] = normalized_text.apply(rule_applies)

    return df

