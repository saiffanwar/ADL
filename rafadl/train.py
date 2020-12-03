import argparse
from pathlib import Path
from typing import NamedTuple, Union

import torch
from torch.utils.tensorboard import SummaryWriter

from coursework import ShallowModel, Trainer
import os

cwd = os.getcwd()
# import torch.backends.cudnn
# torch.backends.cudnn.benchmark = True (from cifar.py, idk what this does)

parser = argparse.ArgumentParser(
    description="Trains shallow CNN on the SALICON Saliency Prediction dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default='')
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=1,
    type=int,
    help="Number of worker processes used to load data.",
)

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
      f"CNN_bn_"
      f"run_"
    )

    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def main(args):

    # create model
    model = ShallowModel()
    print("Created model")

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")

    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    trainer = Trainer(model, args.dataset_root, summary_writer, DEVICE,args.batch_size)
    
    trainer.train(
        args.epochs,
        args.val_frequency,
        log_frequency=args.log_frequency,
    )
    print("Model trained")

    # need to do model saving 
    torch.save(model,"/mnt/storage/home/sa17826/ADL/cw/model.pkl")
    print("Model saved")
    summary_writer.close()

if __name__ == "__main__":
    main(parser.parse_args())
