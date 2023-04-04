import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stochastic", action="store_true", help="stochastic or not")
parser.add_argument("-r", "--robust", action="store_true", help="robust or not")
parser.add_argument("-n", "--num_points", type=int, help="number of sampled points")
parser.add_argument("--exp_dir", type=str, default="exp", help="exp dir")
parser.add_argument("--pred_dir", type=str, default="preds", help="pred dir")
parser.add_argument("--vis_dir", type=str, default="pics", help="vis dir")
parser.add_argument("--seed", type=int, default=131, help="random seed")
opt = parser.parse_args()
if opt.robust:
    opt.exp_dir = opt.exp_dir + "_robust"
print(opt)

set_random_seed(opt.seed)
suffix = "stochastic" if opt.stochastic else "deterministic"


def vis_circle(ax):
    x = np.linspace(-1, 1, 1000)
    upper = np.sqrt(1 - x ** 2)
    lower = -np.sqrt(1 - x ** 2)
    ax.plot(x, upper, c="k")
    ax.plot(x, lower, c="k")
    return ax


def vis_preds(pred_file, opt):
    str_epoch = os.path.splitext(os.path.split(pred_file)[1])[0].split("_")[-1]
    # read
    data = []
    with open(pred_file) as fr:
        for line in fr:
            x1, x2 = line.strip().split()
            data.append([float(x1), float(x2)])

    # sample
    data = random.sample(data, opt.num_points)
    data = np.array(data)

    # draw
    imgpath = os.path.join(opt.exp_dir, opt.vis_dir, f"pred_{suffix}_{str_epoch}.jpg")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots(1, 1)
    vis_circle(ax)
    # in the paper: x2 -> x, x1 -> y
    ax.scatter(data[:, 1], data[:, 0], linewidths=0.3, marker="^", edgecolors=(0, 0, 0))
    ax.grid(True)
    x_lim = np.abs(data[:, 1]).max()
    y_lim = np.abs(data[:, 0]).max()
    lim = max(x_lim, y_lim, 1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.savefig(imgpath)
    print(
        "[Vis {}] [Saved {}]".format(
            os.path.split(pred_file)[1], os.path.split(imgpath)[1],
        )
    )


if __name__ == "__main__":
    os.makedirs(os.path.join(opt.exp_dir, opt.vis_dir), exist_ok=True)
    pred_files = glob.glob(
        os.path.join(opt.exp_dir, opt.pred_dir, f"pred_{suffix}_*.txt")
    )
    for pred_file in sorted(pred_files):
        vis_preds(pred_file, opt)
