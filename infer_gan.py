import argparse
import glob
import os

import numpy as np
import torch
from datasets import build_dataloader
from models import Discriminator, Generator
from torch.autograd import Variable
from utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stochastic", action="store_true", help="stochastic or not")
parser.add_argument("-r", "--robust", action="store_true", help="robust or not")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--exp_dir", type=str, default="exp", help="exp dir")
parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="ckpt dir")
parser.add_argument("--pred_dir", type=str, default="preds", help="pred dir")
parser.add_argument("--seed", type=int, default=131, help="random seed")
opt = parser.parse_args()
if opt.robust:
    opt.exp_dir = opt.exp_dir + "_robust"
print(opt)

set_random_seed(opt.seed)
cuda = True if torch.cuda.is_available() else False
suffix = "stochastic" if opt.stochastic else "deterministic"


def infer(str_epoch):

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Load weights
    G_file = os.path.join(opt.exp_dir, opt.ckpt_dir, f"G_{suffix}_{str_epoch}.pth")
    D_file = os.path.join(opt.exp_dir, opt.ckpt_dir, f"D_{suffix}_{str_epoch}.pth")
    G_state_dict = torch.load(G_file)
    D_state_dict = torch.load(D_file)
    generator.load_state_dict(G_state_dict)
    discriminator.load_state_dict(D_state_dict)

    # Configure data loader
    cfg = {
        "meta_file": "../data/test.txt",
        "batch_size": opt.batch_size,
        "workers": 1,
    }
    dataloader = build_dataloader(cfg)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Infering
    # ----------

    outputs = []
    for i, (x1, _) in enumerate(dataloader):

        # Configure input
        real_x1 = Variable(x1.type(Tensor))

        # -----------------
        #  Generator
        # -----------------

        # Sample noise as generator input
        if opt.stochastic:
            z = Variable(Tensor(np.random.rand(x1.shape[0], 1)))
        else:
            z = Variable(Tensor(np.zeros((x1.shape[0], 1))))

        # Generate a batch
        with torch.no_grad():
            gen_x2 = generator(real_x1, z)

        # Construct outpus
        preds = torch.cat([real_x1, gen_x2], dim=1).cpu().numpy()
        outputs.append(preds)

        print(
            "[Epoch %s] [Batch %d/%d]"
            % (str_epoch.split("epoch")[-1], i + 1, len(dataloader))
        )

    # ---------------------
    #  Save Results
    # ---------------------

    save_txt = os.path.join(opt.exp_dir, opt.pred_dir, f"pred_{suffix}_{str_epoch}.txt")
    with open(save_txt, "w") as fw:
        for batch in outputs:
            for x1, x2 in batch:
                fw.write(f"{x1} {x2}\n")


if __name__ == "__main__":
    os.makedirs(os.path.join(opt.exp_dir, opt.pred_dir), exist_ok=True)
    G_files = glob.glob(os.path.join(opt.exp_dir, opt.ckpt_dir, f"G_{suffix}_*.pth"))
    for G_file in sorted(G_files):
        str_epoch = os.path.splitext(os.path.split(G_file)[1])[0].split("_")[-1]
        infer(str_epoch)
