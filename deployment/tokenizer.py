# tokenizer.py
import sentencepiece as spm
import os, glob, pandas as pd, tempfile
import torch


class IndicSentencePieceTokenizer:
    def __init__(self, vocab_size=64000, model_prefix="indic_tokenizer", model_type="unigram"):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.model_type = model_type
        self.sp = None

    def train(self, files, sample_limit=None):
        print("🧠 Training SentencePiece tokenizer...")
        temp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
        with open(temp_txt, "w", encoding="utf-8") as out:
            for f in files:
                if not os.path.exists(f):
                    continue
                print(f"📘 Reading {f} ...")
                df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip", engine="python")
                for col in df.columns:
                    texts = df[col].astype(str).values[:sample_limit]
                    for text in texts:
                        out.write(text.strip() + "\n")

        print("🧩 Text collection complete. Starting SentencePiece training ...")

        spm.SentencePieceTrainer.Train(
            input=temp_txt,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"]
        )
        print(f"✅ SentencePiece tokenizer trained -> {self.model_prefix}.model")

        os.remove(temp_txt)
        self.sp = spm.SentencePieceProcessor(model_file=f"{self.model_prefix}.model")

    def load(self, model_file="indic_tokenizer.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        print(f"✅ Loaded SentencePiece tokenizer from {model_file}")

    def batch_encode(self, texts, max_length=256):
        ids = [self.sp.encode(t, out_type=int)[:max_length] for t in texts]
        attn_mask = [[1]*len(seq) + [0]*(max_length-len(seq)) if len(seq) < max_length else [1]*max_length for seq in ids]
        ids = [seq + [0]*(max_length-len(seq)) if len(seq) < max_length else seq[:max_length] for seq in ids]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long)
        }

    def __len__(self):
        return len(self.sp)