# from CNN import *
import matplotlib.pyplot as plt
from ShallowCNN import ShallowCNN

parser = argparse.ArgumentParser(
    description="Test a CNN for saliency prediction",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)

### CHECKPOINT ###
parser.add_argument("--resume-checkpoint", type=Path)
parser.add_argument("--preds-path", type=Path, default=Path("preds.pkl"))
parser.add_argument("--filter-path", type=Path, default=Path("conv1_filters.png"))

def main(args):
    model = CNN(height=96, width=96, channels=3).to(DEVICE)

    ### CHECKPOINT - load parameters, args, loss ###
    if args.resume_checkpoint != None and args.resume_checkpoint.exists():
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume_checkpoint)
        else:
            # if CPU is used
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))

        print(f"Testing model {args.resume_checkpoint} that achieved {checkpoint['loss']} loss")
        model.load_state_dict(checkpoint['model'])

    # visualise filters learnt by first conv layer
    weight = model.conv1.weight.data

    # normalise between 0 and 1
    min_val = torch.min(weight)
    max_val = torch.max(weight)
    norm_weight = (weight - min_val)/(max_val - min_val)

    fig, axes = plt.subplots(4,8)
    fig.suptitle("Filters of First Convolutional Layer")

    for i in range(4):
        for j in range(8):
            filter = norm_weight[8*i+j].cpu()
            axes[i, j].imshow(filter.permute(1,2,0))
            axes[i, j].axis('off')
            axes[i, j].set_title(8*i+j+1)

    plt.savefig(args.filter_path, dpi=fig.dpi)
    print(f"Filters of first conv layer saved to {args.filter_path}")

    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    # criterion = lambda logits, labels : torch.mean(torch.sqrt(torch.sum(nn.MSELoss(reduction="none")(logits, labels), dim=1))).requires_grad_(True)
    criterion = nn.MSELoss()

    preds = np.empty([0, 2304])
    total_loss = 0
    model.eval()

    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(batch)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = np.vstack((preds, logits.cpu().numpy()))

    average_loss = total_loss / len(test_loader)

    print(f"validation loss: {average_loss:.5f}")

    # Save predictions to preds.pkl and view it with visualisation/evaluation
    with open(args.preds_path, "wb") as f:
        pickle.dump(preds, f)

    print(f"Saved predictions to {args.preds_path}")

if _name_ == "_main_":
    main(parser.parse_args())