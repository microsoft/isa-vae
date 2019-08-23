# Copyright (c) 2019 Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

def printifpresent(key, values):
    if key in values.keys():
        print('{}: {}'.format(key, values[key]))    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    args = parser.parse_args()

    values = torch.load(args.filename)

    #for key in values.keys():
    #    print(key)
    #    print(values[key])
    #    print('{}: {:.2f}'.format(key, values[key]))

    printifpresent('args', values)
    printifpresent('metric', values)
    printifpresent('logpx', values)
