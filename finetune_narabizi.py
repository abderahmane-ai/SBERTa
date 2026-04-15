"""
Fine-tuning script for NArabizi sentiment classification.
Uses SBERTa's proper architecture:
- Mean pooling over H^(L) (no [CLS] token - SBERTa doesn't use one)
- 3 epochs (matching DziriBERT)
- Standard fine-tuning approach
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm

from sberta.config import SBERTaConfig
from sberta.model import SBERTaModel
from sberta.tokenizer import SBERTaTokenizer


class NArabiziDataset(Dataset):
    """
    NArabizi sentiment dataset.
    Parses CoNLL-U files and matches with sentiment annotations.
    """
    
    def __init__(self, conllu_path, sentiment_path, tokenizer, max_length=128, label2id=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Parse sentiment annotations
        sentiment_dict = {}
        with open(sentiment_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# sent_id'):
                    parts = line.split('\t')
                    if len(parts) == 2:
                        sent_id = parts[0].replace('# sent_id = ', '').strip()
                        label = parts[1].strip()
                        sentiment_dict[sent_id] = label
        
        # Parse CoNLL-U file
        self.examples = []
        current_sent_id = None
        current_text = None
        
        with open(conllu_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# sent_id'):
                    current_sent_id = line.split('=')[1].strip()
                elif line.startswith('# text'):
                    current_text = line.split('=', 1)[1].strip()
                elif not line:  # Empty line = sentence boundary
                    # End of sentence
                    if current_sent_id and current_text and current_sent_id in sentiment_dict:
                        self.examples.append({
                            'sent_id': current_sent_id,
                            'text': current_text,
                            'label': sentiment_dict[current_sent_id]
                        })
                    current_sent_id = None
                    current_text = None
        
        # Create or use provided label mapping
        if label2id is None:
            # Training set: create mapping from scratch
            unique_labels = sorted(set(ex['label'] for ex in self.examples))
            self.label2id = {label: i for i, label in enumerate(unique_labels)}
        else:
            # Dev/test set: use training set's mapping
            self.label2id = label2id
        
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        print(f"Loaded {len(self.examples)} examples")
        print(f"Labels: {self.label2id}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        label = self.label2id[example['label']]
        
        # Tokenise and append [SEP]  (SBERTa has no [CLS]; it uses mean pooling)
        ids = self.tokenizer.encode(text, add_sep=True, max_length=self.max_length)
        
        # Vectorized padding: pre-allocate tensors
        input_ids = torch.full((self.max_length,), self.tokenizer.PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        # Fill with actual values
        length = min(len(ids), self.max_length)
        input_ids[:length] = torch.tensor(ids[:length], dtype=torch.long)
        attention_mask[:length] = 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SBERTaClassifier(nn.Module):
    """
    SBERTa with classification head using mean pooling.
    SBERTa has no [CLS] token - uses mean pooling over H^(L) for sequence-level tasks.
    """
    
    def __init__(self, encoder, num_labels, hidden_size=768, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get encoder outputs
        H, p, s = self.encoder(input_ids, attention_mask)  # (B, T, d)
        
        # Mean pooling: average all token representations, excluding padding
        mask_expanded = attention_mask.unsqueeze(-1).expand(H.size()).float()  # (B, T, d)
        sum_embeddings = torch.sum(H * mask_expanded, dim=1)  # (B, d)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (B, d) - avoid div by zero
        pooled_output = sum_embeddings / sum_mask  # (B, d)
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (B, num_labels)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {"loss": loss, "logits": logits}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate on dev/test set."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "predictions": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data-dir", default="benchmarks/NArabizi/data", help="Path to NArabizi data")
    parser.add_argument("--checkpoint", required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--tokenizer-dir", default="runs/tokenizer", help="Directory containing sberta.model")
    parser.add_argument("--output", required=True, help="Output directory")
    
    # Training (matching DziriBERT)
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (DziriBERT used 3)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (Transformers default)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (Transformers default)")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}\n")
    
    # Load tokenizer
    tokenizer = SBERTaTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"✓ Loaded tokenizer\n")
    
    # Load datasets
    conllu_dir = Path(args.data_dir) / "ud" / "ar_na"
    sentiment_dir = Path(args.data_dir) / "Narabizi" / "sentiment"
    
    print("Loading datasets...")
    train_dataset = NArabiziDataset(
        conllu_dir / "train.conllu",
        sentiment_dir / "train_Narabizi_sentiment.txt",
        tokenizer,
        args.max_length
    )
    
    dev_dataset = NArabiziDataset(
        conllu_dir / "dev.conllu",
        sentiment_dir / "dev_Narabizi_sentiment.txt",
        tokenizer,
        args.max_length,
        label2id=train_dataset.label2id  # Use train's label mapping
    )
    
    test_dataset = NArabiziDataset(
        conllu_dir / "test.conllu",
        sentiment_dir / "test_Narabizi_sentiment.txt",
        tokenizer,
        args.max_length,
        label2id=train_dataset.label2id  # Use train's label mapping
    )
    
    num_labels = len(train_dataset.label2id)
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Dev:   {len(dev_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    print(f"  Labels: {num_labels} classes - {train_dataset.label2id}\n")
    
    # Verify no data leakage
    train_ids = set(ex['sent_id'] for ex in train_dataset.examples)
    dev_ids = set(ex['sent_id'] for ex in dev_dataset.examples)
    test_ids = set(ex['sent_id'] for ex in test_dataset.examples)
    
    assert len(train_ids & dev_ids) == 0, "Data leakage: train and dev overlap!"
    assert len(train_ids & test_ids) == 0, "Data leakage: train and test overlap!"
    assert len(dev_ids & test_ids) == 0, "Data leakage: dev and test overlap!"
    print("✓ No data leakage detected\n")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Load pre-trained model
    config_path = Path(args.checkpoint) / "config.json"
    config = SBERTaConfig.load(config_path)
    
    encoder = SBERTaModel(config)
    checkpoint = torch.load(Path(args.checkpoint) / "model.pt", map_location=device, weights_only=True)
    
    # Load encoder weights (filter out pre-training head weights)
    encoder_state = {
        k.replace("sberta.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("sberta.")
    }
    encoder.load_state_dict(encoder_state)
    print(f"✓ Loaded pre-trained encoder from {args.checkpoint}\n")
    
    # Create classification model
    model = SBERTaClassifier(encoder, num_labels, config.hidden_size, args.dropout)
    model = model.to(device)
    
    # Optimizer (AdamW with weight decay - Transformers default)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        eps=1e-8,
        weight_decay=0.01  # Transformers default
    )
    
    # LR Scheduler (Linear decay with warmup - Transformers default)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("="*80)
    print("TRAINING")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: 0.01")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Total steps: {num_training_steps}")
    print(f"Max length: {args.max_length}")
    print(f"Dropout: {args.dropout}")
    print("="*80 + "\n")
    
    best_dev_acc = 0
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")
        
        dev_results = evaluate(model, dev_loader, device)
        print(f"Dev - Loss: {dev_results['loss']:.4f}, Acc: {dev_results['accuracy']:.4f}, "
              f"F1: {dev_results['f1_macro']:.4f}, Prec: {dev_results['precision_macro']:.4f}, "
              f"Rec: {dev_results['recall_macro']:.4f}")
        
        # Save best model based on dev accuracy
        if dev_results["accuracy"] > best_dev_acc:
            best_dev_acc = dev_results["accuracy"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            
            # Save label mappings for future inference
            import json
            with open(output_dir / "label2id.json", "w") as f:
                json.dump(train_dataset.label2id, f, indent=2)
            with open(output_dir / "id2label.json", "w") as f:
                json.dump({str(k): v for k, v in train_dataset.id2label.items()}, f, indent=2)
            
            print(f"✓ Saved best model (dev acc: {best_dev_acc:.4f})")
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("TEST EVALUATION")
    print("="*80 + "\n")
    
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_results = evaluate(model, test_loader, device)
    
    print(f"Test Accuracy:  {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.1f}%)")
    print(f"Test F1 (macro): {test_results['f1_macro']:.4f}")
    print(f"Test Precision:  {test_results['precision_macro']:.4f}")
    print(f"Test Recall:     {test_results['recall_macro']:.4f}")
    
    # Classification report
    label_names = [train_dataset.id2label[i] for i in range(num_labels)]
    report = classification_report(
        test_results["labels"],
        test_results["predictions"],
        target_names=label_names,
        digits=4
    )
    
    print("\n" + report)
    
    # Save results
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Max length: {args.max_length}\n\n")
        f.write(f"Test Accuracy:  {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.1f}%)\n")
        f.write(f"Test F1 (macro): {test_results['f1_macro']:.4f}\n")
        f.write(f"Test Precision:  {test_results['precision_macro']:.4f}\n")
        f.write(f"Test Recall:     {test_results['recall_macro']:.4f}\n\n")
        f.write(f"DziriBERT benchmark: 63.5%\n")
        f.write(f"SBERTa (ours):       {test_results['accuracy']*100:.1f}%\n")
        f.write(f"Difference:          {(test_results['accuracy']-0.635)*100:+.1f}%\n\n")
        f.write(report)
    
    print(f"\n✓ Results saved to {output_dir}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON TO DZIRIBERT")
    print("="*80)
    print(f"DziriBERT: 63.5%")
    print(f"SBERTa:    {test_results['accuracy']*100:.1f}%")
    
    if test_results['accuracy'] > 0.635:
        print(f"✓ WE WIN! (+{(test_results['accuracy']-0.635)*100:.1f}%)")
    else:
        print(f"✗ We lose (-{(0.635-test_results['accuracy'])*100:.1f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
