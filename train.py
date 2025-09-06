import torch
from torch import nn, optim
from data_loader import get_loaders
from model_unet import UNet

def train_loop(dataloader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_loaders("data/train_images", "data/train_masks", batch_size=4)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()
    
    epochs = 20
    for epoch in range(epochs):
        loss = train_loop(train_loader, model, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), "outputs/unet_vein.pth")

if __name__=="__main__":
    main()
