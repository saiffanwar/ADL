import pickle
from coursework import Salicon, visualise
import torch
from torch.utils.data import DataLoader


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_dataset = Salicon(
            "val.pkl"
        )

    val_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
    )
    modelname = input("Enter a model name: ")
    model = torch.load(modelname)
    preds = []
    total_loss = 0
    model.eval()

    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch, gts in val_loader:
            batch = batch.to(device)
            gts = gts.to(device)
            logits = model(batch)
            outputs = logits.cpu().numpy()
            preds.extend(list(outputs))
    
    with open("val.pkl",'rb') as f:
        val = pickle.load(f)

    visualise(preds,val)

if __name__ == '__main__':
    main()
    