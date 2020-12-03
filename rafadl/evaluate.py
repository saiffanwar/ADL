import pickle
from numpy.core.fromnumeric import mean
from coursework import Salicon, evaluation
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from scipy import ndimage



def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_dataset = Salicon(
            "/mnt/storage/home/sa17826/ADL/cw/val.pkl"
        )

    val_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
    )
    # modelname = input("Enter a model name: ")
    model = torch.load('/mnt/storage/home/sa17826/rafadl/model.pkl')
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
    with open("/mnt/storage/home/sa17826/ADL/cw/val.pkl",'rb') as f:
        val = pickle.load(f)

    print("Made predictions, Loaded GTS")
    
    cc_scores = []
    auc_borji_scores = []
    auc_shuffled_scores = []
    for i in range(len(preds)):
        if i % 10 == 0:
            print(i)
        gt = val[i]['y_original']
        pred = np.reshape(preds[i], (48, 48))
        pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]))
        pred = np.asarray(pred, dtype='float32') / 255.
        pred = ndimage.gaussian_filter(pred, sigma=2)
        cc_scores.append(evaluation.cc(pred, gt))
        auc_borji_scores.append(evaluation.auc_borji(pred, np.asarray(gt, dtype=np.int)))

        # Sample 10 random fixation maps for AUC Shuffled and take their union
        other = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.int)
        randind_maps = np.random.choice(len(val), size=10, replace=False)
        # this is a huge antipattern
        for i in range(10):
            other = other | np.asarray(val[randind_maps[i]]['y_original'], dtype=np.int)

        auc_shuffled_scores.append(evaluation.auc_shuff(pred, np.asarray(gt, dtype=np.int), other))

    # CC
    print('CC: {}'.format(np.mean(cc_scores)))
    # AUC Borji
    print('AUC Borji: {}'.format(np.mean(auc_borji_scores)))
    # Shuffled AUC
    print('AUC Shuffled: {}'.format(np.mean(auc_shuffled_scores)))
   

if __name__ == '__main__':
    main()
    