import pandas as pd
import numpy as np

import spacy
nlp = spacy.load("it_core_news_sm")

import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

np.random.seed(1234)

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def clean_diagnosi(d):
  words = ['esito:', 'decorso clinico', 'consigli terapeutici', 'consiglio', 'controllo', 'a domicilio']
  for w in words:
    if w in d:
      i = d.index(w)
      d = d[:i]
  if '. ' in d:
    i = d.index('. ')
    d = d[:i]
  d = re.sub(r'paziente: \( id: \d+ \)', '', d)
  return d

def extract_diagnosi(text):
  s = text.lower()
  s = re.sub('[ ]+', ' ', s)
  m = None
  for mm in re.finditer(r'diagnosi\s?:', s, flags=re.IGNORECASE):
    m = mm
  if m is None:
    for mm in re.finditer(r'diagnosi( di dimissione| testuale| alla dimissione)\s?:', s, flags=re.IGNORECASE):
      m = mm
    if m is None:
      for mm in re.finditer(r'diagnosi( di dimissione| testuale| alla dimissione)', s, flags=re.IGNORECASE):
        m = mm
      if m is None:
        for mm in re.finditer(r'diagnosi', s, flags=re.IGNORECASE):
          m = mm
  if m is not None:
    i = m.span()[1]
    s = s[i:]
    #i = s.index('\n')
    #if i < 3:
    s = s.strip()
    i = find_nth(s, '\n', 1)
    s = s[:i]
    s = clean_diagnosi(s)
    s = s.strip()
    if s == '':
      s = None
    return s