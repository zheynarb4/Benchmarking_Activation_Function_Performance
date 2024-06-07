import torch
import torch.nn as nn
import argparse
import config
import dataloader
from tqdm.auto import tqdm
import datetime
import os
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.CrossEntropyLoss()



def train(model, name):
    best_valid_acc = 0
    train_loss_epochs = []
    train_acc_epochs = []
    valid_loss_epochs = []
    valid_acc_epochs = []
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = []
        train_accs = []

        for batch in tqdm(dataloader.train_loader):
            optimizer.zero_grad()
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim = -1) == labels).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc.item())
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        
        train_loss_epochs.append(train_loss)
        train_acc_epochs.append(train_acc)

        model.eval()

        valid_loss = []
        valid_accs = []

        for batch in tqdm(dataloader.valid_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(imgs)

            loss = criterion(logits, labels)
            acc = (logits.argmax(dim = -1) == labels).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc.item())

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        valid_loss_epochs.append(valid_loss)
        valid_acc_epochs.append(valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            current_time = datetime.datetime.now().strftime("%Y-%m-%d")
            file_path = os.path.join(config.MODEL_DIR, f"{name}_{current_time}.pth")
            torch.save(model.state_dict(), file_path)
        
        df = pd.DataFrame(data = {
            "train_loss_epochs": train_loss_epochs,
            "train_acc_epochs": train_acc_epochs,
            "valid_loss_epochs": valid_loss_epochs,
            "valid_acc_epochs": valid_acc_epochs
        })

        df.to_pickle(f"./metrics/{name}_train_loss.pkl")
        
        print(f"[Epoch: {epoch + 1:03d}/{config.EPOCHS:03d}] Train: loss={train_loss:.5f}, acc={train_acc:.5f} | Valid: loss={valid_loss:.5f}, acc={valid_acc:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "train",
        description = "Trains a model with a specific Activation Function",
        epilog = "Happy Training"
    )
    parser.add_argument("architecture", nargs = "?",
                        default = "sigmoid",
                        help = "Takes the model Activation Function (sigmoid, relu, selu) default = sigmoid")
    args = parser.parse_args()
    model = config.MODELS.get(args.architecture)
    train(model, args.architecture)