# Copyright (c) 2019 Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()

    values = torch.load(args.filename)

    print('MIG: {:.2f}'.format(values['metric']))
