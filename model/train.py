import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, BertTokenizer
from tqdm import tqdm

from model import PathVQAModel
from dataset import PathVQADataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
SAVE_PATH = "best_model.pth"


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    print(f"Device: {DEVICE}")

    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_ds = PathVQADataset("train", image_processor, tokenizer)
    val_ds = PathVQADataset("validation", image_processor, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = PathVQAModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch} | train_loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  Saved best model (val_acc={best_acc:.4f})")

    print(f"Training complete. Best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
