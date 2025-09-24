# image_train.py
"""
Treinamento PyTorch profissional:
- Requisitos: torch, torchvision, tqdm
pip install torch torchvision tqdm
Exemplo de uso: estruture pastas:
data/train/<class>/*.jpg
data/val/<class>/*.jpg
"""

import os
import time
import copy
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)

def create_dataloaders(data_dir, batch_size=32, img_size=224):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)

def build_model(num_classes, pretrained=True, freeze_backbone=True):
    model = models.resnet50(pretrained=pretrained)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model.to(DEVICE)

def train_loop(model, train_loader, val_loader, epochs=10, lr=1e-3, out_dir='checkpoints'):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_corrects += torch.sum(preds == labels.data).item()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        scheduler.step(val_acc)
        print(f"Epoch {epoch} Train loss {epoch_loss:.4f} acc {epoch_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

        # checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': val_acc
        }
        torch.save(ckpt, f"{out_dir}/ckpt_epoch_{epoch}.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"{out_dir}/best_model.pth")
            print("==> New best model saved")

    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Diretório com subpastas train/ val/')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    train_loader, val_loader, n_classes = create_dataloaders(args.data, batch_size=args.batch)
    model = build_model(n_classes, pretrained=True, freeze_backbone=True)
    model = train_loop(model, train_loader, val_loader, epochs=args.epochs)
    print("Treino finalizado. Modelo em memória e checkpoints salvos.")
