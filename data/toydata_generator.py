import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=131, help="random seed")
opt = parser.parse_args()
print(opt)


class ToydataGenerator:
    """
    Generate (x, y) with x ** 2 + y ** 2 <= 1
    """

    def __init__(self, num_points, out_txt):
        self.num_points = num_points
        self.out_txt = out_txt
        self.data = self.generate(num_points)
        self.write()

    def generate(self, num_points):
        # 4.0: area of |x| <= 1, |y| <= 1
        # pi: area of x** 2 + y ** 2 <= 1
        scale = 4.0 / np.pi
        num_generate = int(np.ceil(num_points * scale))
        x = np.random.rand(num_generate) * 2 - 1  # [-1, 1] uniform
        y = np.random.rand(num_generate) * 2 - 1  # [-1, 1] uniform
        mask = x ** 2 + y ** 2 <= 1
        x, y = x[mask], y[mask]
        if len(x) >= num_points:
            x, y = x[:num_points], y[:num_points]
        else:
            x_add, y_add = self.generate(num_points - len(x))
            x = np.concatenate([x, x_add], axis=0)
            y = np.concatenate([y, y_add], axis=0)
        return x, y

    def write(self):
        with open(self.out_txt, "w") as fw:
            for x, y in zip(*self.data):
                fw.write(f"{x} {y}\n")


if __name__ == "__main__":
    np.random.seed(opt.seed)

    train_points = 100000
    train_txt = "./train.txt"
    train_generator = ToydataGenerator(train_points, train_txt)

    test_points = 10000
    test_txt = "./test.txt"
    test_generator = ToydataGenerator(test_points, test_txt)
