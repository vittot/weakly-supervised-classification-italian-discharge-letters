import re
import torch
import pandas as pd
import numpy as np

to_clean = [
#'regione',
#'azienda',
'verbale',
'pag.',
'pag ',
'pagina',
'nato',
'tess.san',
'codice fiscale',
'comune di nascita',
'cap',
'indirizzo',
'cartella dea',
'documento firmato digitalmente',
'informazione ai sensi',
'il medico dimettente',
'gentile signor',
'Copia di documento firmato e conservato',
'desideriamo renderLa partecipe',
#'U.L.S.S.',
'modulo di pronto soccorso',
'direttore',
'ai genitori',
'al medico',
'indirizzo',
'residente',
'residenza',
#'diagnosi',
'infermier',
'nome',
'cognome',
'firma',
'consegnare al proprio pediatra',
"l'orario di alcune prestazioni",
"l' orario di alcune prestazioni",
#'dipartimento',
      'verbale di pronto soccorso',
      'della cartella',
      'modulo di',
      'numero di certificato',
      'firmatario',
      'il referto e\' conservato',
      'ID Documento',
      'Gentile signore',
      'Informazione',
      'dettagli paziente',
      'verbale n',
      'cartella DEA',
      'priorita',
      'tel',
      'fax',
      'indirizzo',
      'residenza',
      'domicilio',
      'residente',
      'colleghi',
      'collega',
      'segreteria',
      'data e ora'
]

def clean_sentences(doc, clean_diagnosi=True):
  doc = doc.replace('\t', '\n')
  doc = re.sub(r'    [ ]*', ' ', doc)
  sentences = doc.split('\n')
  cleaned_sentences = []
  to_clean_2 = to_clean.copy()
  if clean_diagnosi:
    to_clean_2 = to_clean_2 + ['diagnosi']
  for s in sentences:
    if not any(ss.lower() in s.lower() for ss in to_clean_2) and len(s) > 10:
      cleaned_sentences.append(s.strip())
  res =  '\n'.join(cleaned_sentences)
  res = re.sub(r'\d{2}\.\d{2}\.\d{2,4}', '', res) # dates
  res = re.sub(r'\d{2}/\d{2}/\d{2,4}', '', res) # dates
  res = re.sub(r'\d{2}:\d{2}', '', res) # hours
  return res

def _get_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = "cpu"
    return torch.device(dev)

def sample_balanced_fse(df, text_col, label_col, our_label_col):
    unique_ys = df[our_label_col].unique()
    X_res = pd.DataFrame(columns=[text_col, label_col, our_label_col, 'localita', 'azienda', 'pediatria'])
    n_samples = min(df.groupby(our_label_col).size())
    for unique_y in unique_ys:
        val_indices = df[df[our_label_col]==unique_y].index
        random_samples = np.random.choice(val_indices, n_samples, replace=False)
        X_res = pd.concat([X_res, df.loc[random_samples, [text_col, label_col, our_label_col, 'pediatria', 'localita', 'azienda']]])
    X_res
    return X_res