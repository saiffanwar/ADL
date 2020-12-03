import argparse
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
from scipy import ndimage
import matplotlib.pyplot as plt
import torch
from dataset import Salicon
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Visualising model outputs')

# parser.add_argument('--preds', help='Model predictions')
# parser.add_argument('--gts', help = 'Ground truth data')
parser.add_argument('--outdir', default = '.', type=Path, help='output directory for visualisation')

args = parser.parse_args()

def main():
    #loading preds and gts
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_dataset = Salicon(
            # "/mnt/storage/home/sa17826/ADL/cw/val.pkl"
            "val.pkl"

        )

    val_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
    )
    # modelname = input("Enter a model name: ")
    # model = torch.load(modelname)
    model = torch.load('/mnt/storage/home/sa17826/ADL/cw/model.pkl')
    total_loss = 0
    model.eval()
    preds = []

    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch, gts in val_loader:
            batch = batch.to(device)
            gts = gts.to(device)
            logits = model(batch)
            outputs = logits.cpu().numpy()
            preds.extend(list(outputs))
    # with open("/mnt/storage/home/sa17826/ADL/cw/val.pkl",'rb') as f:
    with open("val.pkl",'rb') as f:

        gts = pickle.load(f)

    print("Made predictions, Loaded GTS")

    index = np.random.randint(0, len(preds), size=2) #get indices for 3 random images

    outputs = []
    for idx in index:
        #getting original image
        image = gts[idx]['X_original']
        image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
        outputs.append(image)

        #getting ground truth saliency map
        sal_map = gts[idx]['y_original']
        sal_map = ndimage.gaussian_filter(sal_map, 19)
        outputs.append(sal_map)

        #getting model prediction
        pred = np.reshape(preds[idx], (48, 48))
        pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]))
        pred = np.asarray(pred, dtype='float32') / 255.
        pred = ndimage.gaussian_filter(pred, sigma=2)
        # outputs.append(pred)

    #plotting images 
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(32,32))
    ax[0][0].set_title("Image", fontsize=75)
    ax[0][1].set_title("Ground Truth", fontsize=75)
    # ax[0][2].set_title("Prediction", fontsize=40)
    
    fig.tight_layout()

    for i, axi in enumerate(ax.flat):
        axi.imshow(outputs[i])
    
    #saving output
    if not args.outdir.parent.exists():
        args.outdir.parent.mkdir(parents=True)
    outpath = os.path.join(args.outdir, "truths.pdf")
    plt.savefig(outpath)
    
if __name__ == '__main__':
    main()