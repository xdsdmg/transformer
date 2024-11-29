import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import data
import model


if __name__ == "__main__":
    data_set = data.DataSet()

    # m=model.Tra

    for i in range(data_set.__len__()):
        print(data_set.__getitem__(i))
