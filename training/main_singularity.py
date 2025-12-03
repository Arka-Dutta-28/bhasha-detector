# main_singularity.py
import os
import json
import time
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tokenizer import IndicSentencePieceTokenizer
from model import TransformerEncoder, TripletLoss
from dataset import (
    get_indic_processor,
    build_triplet_dataloaders,
    build_phase2_dataloader,
    safe_read_csv
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# CONFIG
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "train.log"
LOG_INTERVAL = 1000  # log every 1000 batches


# ===============================
# Utility Logging
# ===============================
def log(message):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    msg = f"{timestamp} {message}"
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)
    log(f"✅ Saved checkpoint -> {path}")


# ===============================
# Phase 1 Training
# ===============================
def train_phase1(model, optimizer, tokenizer, loader, epochs=5, save_dir="checkpoints"):
    model.train()
    triplet_loss = TripletLoss(margin=0.5)
    log(f"🚀 Starting Phase 1 training for {epochs} epochs")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader, start=1):
            a_ids = batch['anchor_ids'].to(DEVICE)
            p_ids = batch['pos_ids'].to(DEVICE)
            n_ids = batch['neg_ids'].to(DEVICE)
            a_mask = batch['anchor_mask'].to(DEVICE)
            p_mask = batch['pos_mask'].to(DEVICE)
            n_mask = batch['neg_mask'].to(DEVICE)

            emb_a = model(a_ids, a_mask)
            emb_p = model(p_ids, p_mask)
            emb_n = model(n_ids, n_mask)
            loss = triplet_loss(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % LOG_INTERVAL == 0:
                avg = total_loss / batch_idx
                log(f"[Phase1][Epoch {epoch}] Batch {batch_idx}/{len(loader)} | Loss={loss.item():.8f} | Avg={avg:.8f}")

        avg_epoch = total_loss / max(1, len(loader))
        elapsed = (time.time() - start_time) / 60
        log(f"✅ Phase1 Epoch {epoch} done | AvgLoss={avg_epoch:.8f} | Time={elapsed:.2f} min")
        
        # Step scheduler based on the average loss of the epoch
        scheduler.step(avg_epoch)
        
        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f"phase1_epoch{epoch}.pt"))



# ===============================
# Phase 2 Training
# ===============================
def train_phase2(model, optimizer, tokenizer, loader, epochs=5, save_dir="checkpoints"):
    model.train()
    criterion = nn.CrossEntropyLoss()
    log(f"🚀 Starting Phase 2 training for {epochs} epochs")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=0)

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader, start=1):
            ids = batch['input_ids'].to(DEVICE)
            masks = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits, _ = model(ids, masks)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            if batch_idx % LOG_INTERVAL == 0:
                acc = correct / total
                avg = total_loss / batch_idx
                log(f"[Phase2][Epoch {epoch}] Batch {batch_idx}/{len(loader)} | Loss={loss.item():.8f} | Avg={avg:.8f} | Acc={acc:.8f}")

        avg_loss = total_loss / max(1, len(loader))
        avg_acc = correct / max(1, total)
        elapsed = (time.time() - start_time) / 60
        log(f"✅ Phase2 Epoch {epoch} done | AvgLoss={avg_loss:.8f} | Acc={avg_acc:.8f} | Time={elapsed:.8f} min")
        
        # Step scheduler based on the average loss of the epoch
        scheduler.step(avg_loss)
        
        save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f"phase2_epoch{epoch}.pt"))



# ===============================
# Evaluation
# ===============================

def evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv="bhasha-abhijnaanam.csv", label_map_path="label2id.json", max_len=256, device='cuda'):
    model.eval()
    df = pd.read_csv(bhasha_csv, encoding="utf-8", on_bad_lines="skip", engine="python")
    
    with open(label_map_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {int(v): k for k, v in label2id.items()}
    y_true, y_pred = [], []
    print(f"🔍 Starting evaluation on {len(df)} samples")
    all_preds = []

    with torch.no_grad():
        for i, row in enumerate(df.itertuples(), start=1):
            text = processor.process(row.text, getattr(row, 'label', None))
            enc = tokenizer.batch_encode([text], max_length=max_len)
            ids = enc['input_ids'].to(device)
            masks = enc['attention_mask'].to(device)
            
            logits, _ = model(ids, masks)
            pred = int(torch.argmax(logits, dim=-1).item())
            
            gold_name = getattr(row, 'label', None)
            gold = label2id.get(gold_name, None)
            if gold is None:
                continue
            
            y_true.append(gold)
            y_pred.append(pred)
            all_preds.append(pred)
            
            if i % 1000 == 0:
                print(f"Eval progress: {i}/{len(df)} samples...")
   
    acc = accuracy_score(y_true, y_pred) * 100
    report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)])
    print(f"✅ Evaluation complete | Accuracy={acc:.4f}%")
    
    with open("bhasha_eval_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}%\n\n{report}")
    print("📄 Saved report -> bhasha_eval_report.txt")
    
    cm = confusion_matrix(y_true, y_pred)
    labels = [id2label[i] for i in sorted(id2label)]
    with open("bhasha_eval_report.txt", "a", encoding="utf-8") as f:
        f.write("\n\n=== Confusion Matrix ===\n")
        f.write("\t" + "\t".join(labels) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{labels[i]}\t" + "\t".join(map(str, row)) + "\n")
    print("📄 Appended confusion matrix to bhasha_eval_report.txt")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix - Bhasha Evaluation")
    plt.tight_layout()
    plt.savefig("bhasha_confusion_matrix.png", dpi=300)
    plt.close()
    print("🖼️ Saved confusion matrix plot -> bhasha_confusion_matrix.png")



# ===============================
# Main
# ===============================
def main():
    processor = get_indic_processor()

    tokenizer = IndicSentencePieceTokenizer(vocab_size=32000)
    if not os.path.exists("indic_tokenizer.model"):
        log("⚙️ Training SentencePiece tokenizer...")
        tokenizer.train(["phase1.csv", "phase2.csv"])
    else:
        tokenizer.load("indic_tokenizer.model")
        log("✅ Loaded existing SentencePiece tokenizer")

    # ========== PHASE 1 ==========
    loader_nre = build_triplet_dataloaders("phase1.csv", processor, tokenizer, batch_size=32)
    model = TransformerEncoder(
        vocab_size=len(tokenizer),
        embed_dim=256,
        num_layers=2,
        num_heads=8,
        ff_dim=1024,
        phase="phase1"
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_phase1(model, optimizer, tokenizer, loader_nre, epochs=2)
    save_checkpoint(model, optimizer, 1, "checkpoints/phase1_final.pt")

    # ========== PHASE 2 ==========
    model.phase = "phase2"
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 22)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    loader_p2 = build_phase2_dataloader("phase2.csv", processor, tokenizer, batch_size=32)
    train_phase2(model, optimizer, tokenizer, loader_p2, epochs=2)
    save_checkpoint(model, optimizer, 2, "checkpoints/phase2_final.pt")

    # ========== EVALUATION ==========
    evaluate_on_bhasha(model, tokenizer, processor, bhasha_csv="bhasha-abhijnaanam.csv", label_map_path="label2id.json")


#def main():
#    processor = get_indic_processor()
#
#    tokenizer = IndicSentencePieceTokenizer(vocab_size=32000)
#    if not os.path.exists("indic_bpe_tokenizer.model"):
#        log("⚙️ Training SentencePiece tokenizer...")
#        tokenizer.train(["phase1.csv", "phase2.csv"])
#    else:
#        tokenizer.load("indic_bpe_tokenizer.model")
#        log("✅ Loaded existing SentencePiece tokenizer")
#
#    # ======================
#    #  PHASE 1 (train once)
#    # ======================
#    loader_nre = build_triplet_dataloaders("phase1.csv", processor, tokenizer, batch_size=32)
#
#    model = TransformerEncoder(
#        vocab_size=len(tokenizer),
#        embed_dim=256,
#        num_layers=6,
#        num_heads=8,
#        ff_dim=1024,
#        phase="phase1"
#    ).to(DEVICE)
#
#    # ---- Train Phase 1 ----
#    # Uncomment if Phase 1 is not trained yet
#    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#    # train_phase1(model, optimizer, tokenizer, loader_nre, epochs=2)
#    # save_checkpoint(model, optimizer, 2, "checkpoints/phase1_final.pt")
#
#    # ======================
#    # LOAD PHASE 1 WEIGHTS
#    # ======================
#    ckpt_path = "checkpoints/phase1_final.pt"
#    assert os.path.exists(ckpt_path), "❌ Phase1 checkpoint missing!"
#
#    ckpt = torch.load(ckpt_path, map_location=DEVICE)
#    model.load_state_dict(ckpt["model_state"], strict=False)
#    log("✅ Loaded Phase-1 checkpoint")
#
#    # =========================================
#    #  PREPARE MODEL FOR PHASE 2 TRAINING
#    # =========================================
#    model.phase = "phase2"
#    model.classifier = nn.Sequential(
#        nn.Dropout(0.2),
#        nn.Linear(256, 22)
#    ).to(DEVICE)
#
#    # NEW Learning Rate for Phase 2
#    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
#    log("🔥 Starting Phase-2 with NEW LR = 4e-5")
#
#    loader_p2 = build_phase2_dataloader("phase2.csv", processor, tokenizer, batch_size=32)
#
#    # ---- Train Phase 2 ----
#    train_phase2(model, optimizer, tokenizer, loader_p2, epochs=2)
#    save_checkpoint(model, optimizer, 2, "checkpoints/phase2_final.pt")
#
#    # ======================
#    #        EVAL
#    # ======================
#    evaluate_on_bhasha(
#        model, tokenizer, processor,
#        bhasha_csv="bhasha-abhijnaanam.csv",
#        label_map_path="label2id.json"
#    )


if __name__ == "__main__":
    log("======== TRAINING STARTED ========")
    log(f"Device: {DEVICE}")
    main()
    log("======== TRAINING COMPLETED ========")