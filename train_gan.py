import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from datasets import build_dataloader
from loss import RobustLoss
from models import Discriminator, Generator, weights_init_normal
from torch.autograd import Variable
from utils import set_random_seed

parser = argparse.ArgumentParser()
# main configs
parser.add_argument("-s", "--stochastic", action="store_true", help="stochastic or not")
# robust configs
parser.add_argument("-r", "--robust", action="store_true", help="robust or not")
parser.add_argument(
    "--weight_robust", type=float, default=0.1, help="weight of robust loss"
)
# training configs
parser.add_argument("--epochs", type=int, default=100, help="number epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument(
    "--b1", type=float, default=0, help="adam: decay of 1st order momentum"
)
parser.add_argument(
    "--b2", type=float, default=0.9, help="adam: decay of 2rd order momentum"
)
# other configs
parser.add_argument("--exp_dir", type=str, default="exp", help="exp dir")
parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="ckpt dir")
parser.add_argument(
    "--save_interval", type=int, default=5, help="epoch interval to save"
)
parser.add_argument("--seed", type=int, default=131, help="random seed")
opt = parser.parse_args()
if opt.robust:
    opt.exp_dir = opt.exp_dir + "_robust"
print(opt)

set_random_seed(opt.seed)
cuda = True if torch.cuda.is_available() else False
suffix = "stochastic" if opt.stochastic else "deterministic"
os.makedirs(os.path.join(opt.exp_dir, opt.ckpt_dir), exist_ok=True)

# Loss function
adversarial_loss = nn.BCELoss()
if opt.robust:
    robust_lost = RobustLoss(
        opt.stochastic, num_max_sample=10, num_z_sample=50, epsilon=0.001
    )

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
cfg = {
    "meta_file": "./data/train.txt",
    "batch_size": opt.batch_size,
    "workers": 1,
}
dataloader = build_dataloader(cfg)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.epochs):
    for i, (x1, x2) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(x1.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(x1.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_x1 = Variable(x1.type(Tensor))
        real_x2 = Variable(x2.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        if opt.stochastic:
            z = Variable(Tensor(np.random.rand(x1.shape[0], 1)))
        else:
            z = Variable(Tensor(np.zeros((x1.shape[0], 1))))

        # Generate a batch of images
        gen_x2 = generator(real_x1, z)

        # Loss measures generator's ability to fool the discriminator
        if opt.robust:
            g_loss = adversarial_loss(
                discriminator(real_x1, gen_x2), valid
            ) + opt.weight_robust * robust_lost(generator, real_x1)
        else:
            g_loss = adversarial_loss(discriminator(real_x1, gen_x2), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_x1, real_x2), valid)
        fake_loss = adversarial_loss(discriminator(real_x1, gen_x2.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (
                epoch + 1,
                opt.epochs,
                i + 1,
                len(dataloader),
                d_loss.item(),
                g_loss.item(),
            )
        )

    # ---------------------
    #  Save Model
    # ---------------------

    if (epoch + 1) % opt.save_interval == 0 or epoch + 1 < 5:
        G_file = os.path.join(
            opt.exp_dir, opt.ckpt_dir, f"G_{suffix}_epoch{epoch + 1}.pth"
        )
        D_file = os.path.join(
            opt.exp_dir, opt.ckpt_dir, f"D_{suffix}_epoch{epoch + 1}.pth"
        )
        torch.save(generator.state_dict(), G_file)
        torch.save(discriminator.state_dict(), D_file)
