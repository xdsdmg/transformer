import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_

raw_data = [
    # [encoder_input, decoder_input, decoder_output]
    ["ich mochte ein bier P", "S i want a beer .", "i want a beer . E"],
    ["ich mochte ein cola P", "S i want a coke .", "i want a coke . E"],
]

src_vocab = {"P": 0, "ich": 1, "mochte": 2, "ein": 3, "bier": 4, "cola": 5}
src_vocab_size = len(src_vocab)

target_vocab = {"P": 0, "i": 1, "want": 2, "a": 3, "beer": 4, "coke": 5, "S": 6, "E": 7, ".": 8}
target_vocab_size = len(target_vocab)

src_len = max([len(example[0].split()) for example in raw_data])
target_len = max([len(example[1].split()) for example in raw_data])


def make_data(raw_data: list[list[str]]) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    encoder_inputs, decoder_inputs, decoder_outputs = [], [], []

    for i in range(len(raw_data)):
        encoder_input = [src_vocab[word] for word in raw_data[i][0].split()]
        decoder_input = [target_vocab[word] for word in raw_data[i][1].split()]
        decoder_output = [target_vocab[word] for word in raw_data[i][2].split()]

        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        decoder_outputs.append(decoder_output)

    return torch.LongTensor(encoder_inputs), torch.LongTensor(decoder_inputs), torch.LongTensor(decoder_outputs)


if __name__ == "__main__":
    print(make_data(raw_data))
