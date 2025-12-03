# dataset.py
import os
import json
import unicodedata
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from typing import Optional
import csv, sys, chardet

# Increase CSV field limit for long Indic texts
csv.field_size_limit(sys.maxsize)

# language -> indicnlp code mapping
lang_code_map = {
    'Assamese': 'as', 'Bodo': 'brx', 'Bangla': 'bn', 'Konkani': 'gom',
    'Gujarati': 'gu', 'Hindi': 'hi', 'Kannada': 'kn', 'Maithili': 'mai',
    'Malayalam': 'ml', 'Marathi': 'mr', 'Nepali': 'ne', 'Oriya': 'or',
    'Punjabi': 'pa', 'Sanskrit': 'sa', 'Sindhi': 'sd', 'Tamil': 'ta',
    'Telugu': 'te', 'Urdu': 'ur', 'Kashmiri': 'ks', 'Manipuri': 'mni',
    'Dogri': 'doi', 'Santali': 'sat'
}


# ----------------------------------------------------------
# Utility: Safe CSV Reader
# ----------------------------------------------------------
def safe_read_csv(path, sample_limit=None, max_chars_per_field=5000):
    """Robust CSV reader for Indic data."""
    with open(path, 'rb') as f:
        raw = f.read(100_000)
        enc_guess = chardet.detect(raw)
        encoding = enc_guess['encoding'] or 'utf-8'

    try:
        df = pd.read_csv(path, encoding=encoding, on_bad_lines='skip',
                         engine='python', nrows=sample_limit)
    except Exception as e:
        print(f"⚠️ Pandas failed ({type(e).__name__}): {e}")
        print("Retrying with manual CSV reader...")
        rows = []
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                rows.append(row)
        df = pd.DataFrame(rows)
        df = df.rename(columns=df.iloc[0]).drop(df.index[0], errors='ignore')

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).apply(lambda x: x[:max_chars_per_field])
    return df


# ----------------------------------------------------------
# Text Processor
# ----------------------------------------------------------
def fallback_normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


class IndicTextProcessor:
    def __init__(self):
        self.factory = IndicNormalizerFactory()
        self._cache = {}

    def process(self, text: str, label: Optional[str] = None):
        text = str(text).strip()
        if not text:
            return text
        code = lang_code_map.get(label, 'hi')
        try:
            self._cache = {}
            if code not in self._cache:
                self._cache[code] = self.factory.get_normalizer(code)
            norm = self._cache[code].normalize(text)
            toks = indic_tokenize.trivial_tokenize(norm, code)
            return " ".join(toks)
        except Exception:
            return fallback_normalize(text)

# class IndicTextProcessor:
#     def __init__(self):
#         self.factory = IndicNormalizerFactory()
#
#     def process(self, text: str, label: Optional[str] = None):
#         text = str(text).strip()
#         if not text:
#             return text
#         code = lang_code_map.get(label, 'hi')
#         try:
#             normalizer = self.factory.get_normalizer(code)
#             norm = normalizer.normalize(text)
#             toks = indic_tokenize.trivial_tokenize(norm, code)
#             return " ".join(toks)
#         except Exception:
#             return fallback_normalize(text)

# ----------------------------------------------------------
# Triplet Dataset (NRE only)
# ----------------------------------------------------------
class TripletDataset(Dataset): 
    def __init__(self, csv_path, processor: IndicTextProcessor):
        self.df = safe_read_csv(csv_path)
        self.processor = processor
        required = {'anchor', 'positive', 'negative'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a, p, n = row['anchor'], row['positive'], row['negative']
        a = self.processor.process(a)
        p = self.processor.process(p)
        n = self.processor.process(n)
        return a, p, n


# ----------------------------------------------------------
# Phase 2 Dataset
# ----------------------------------------------------------
class Phase2Dataset(Dataset):
    def __init__(self, csv_path, processor: IndicTextProcessor, label_map_path='label2id.json'):
        self.df = safe_read_csv(csv_path)
        self.processor = processor
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.label2id = json.load(f)
        else:
            langs = sorted(self.df['label'].unique())
            self.label2id = {lang: i for i, lang in enumerate(langs)}
            with open(label_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.label2id, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.processor.process(row['text'], row['label'])
        label_id = self.label2id[row['label']]
        return text, label_id


# ----------------------------------------------------------
# Collate Functions
# ----------------------------------------------------------


def collate_triplet_batch(batch, tokenizer, max_len=256):
    anchors, positives, negatives = zip(*batch)
    a_enc = tokenizer.batch_encode(anchors, max_length=max_len)
    p_enc = tokenizer.batch_encode(positives, max_length=max_len)
    n_enc = tokenizer.batch_encode(negatives, max_length=max_len)
    return {
        'anchor_ids': a_enc['input_ids'], 'anchor_mask': a_enc['attention_mask'],
        'pos_ids': p_enc['input_ids'], 'pos_mask': p_enc['attention_mask'],
        'neg_ids': n_enc['input_ids'], 'neg_mask': n_enc['attention_mask'],
    }



def collate_phase2_batch(batch, tokenizer, max_len=256):
    texts, labels = zip(*batch)
    enc = tokenizer.batch_encode(texts, max_length=max_len)
    return {
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'labels': torch.tensor(labels, dtype=torch.long)
    }



# ----------------------------------------------------------
# Builders
# ----------------------------------------------------------
# dataloader
def build_triplet_dataloaders(nre_csv, processor, tokenizer, batch_size=32, num_workers=4):
    ds_nre = TripletDataset(nre_csv, processor)
    loader_nre = DataLoader(
        ds_nre,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_triplet_batch(b, tokenizer)
    )
    return loader_nre



def build_phase2_dataloader(csv_path, processor, tokenizer, batch_size=64, num_workers=4):
    ds = Phase2Dataset(csv_path, processor)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_phase2_batch(b, tokenizer)
    )
    return loader



def get_indic_processor():
    return IndicTextProcessor()
