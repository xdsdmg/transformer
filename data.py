import torch
import torch.utils.data as data_

"""
RAW_DATA:
[[the input of encoder, the input of decoder, the output of decoder], ...]
"""
RAW_DATA = [
    ["ich mochte ein bier P", "S i want a beer .", "i want a beer . E"],
    ["ich mochte ein cola P", "S i want a coke .", "i want a coke . E"],
]

SRC_VOCAB = {"P": 0, "ich": 1, "mochte": 2, "ein": 3, "bier": 4, "cola": 5}
SRC_VOCAB_SIZE = len(SRC_VOCAB)

TGT_VOCAB = {
    "P": 0,
    "i": 1,
    "want": 2,
    "a": 3,
    "beer": 4,
    "coke": 5,
    "S": 6,
    "E": 7,
    ".": 8,
}
TGT_VOCAB_SIZE = len(TGT_VOCAB)

src_len = [len(example[0].split()) for example in RAW_DATA]
tgt_len = [len(example[1].split()) for example in RAW_DATA]


def make_data(
        raw_data: list[list[str]],
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    encoder_inputs, decoder_inputs, decoder_outputs = [], [], []

    for i in range(len(raw_data)):
        encoder_input = [SRC_VOCAB[word] for word in raw_data[i][0].split()]
        decoder_input = [TGT_VOCAB[word] for word in raw_data[i][1].split()]
        decoder_output = [TGT_VOCAB[word] for word in raw_data[i][2].split()]

        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        decoder_outputs.append(decoder_output)

    return (
        torch.LongTensor(encoder_inputs),
        torch.LongTensor(decoder_inputs),
        torch.LongTensor(decoder_outputs),
    )


class DataSet(data_.Dataset):
    def __init__(self):
        super(DataSet, self).__init__()
        encoder_inputs, decoder_inputs, decoder_outputs = make_data(RAW_DATA)
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_outputs = decoder_outputs

    def __len__(self) -> int:
        return self.encoder_inputs.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.encoder_inputs[idx],
            self.decoder_inputs[idx],
            self.decoder_outputs[idx],
        )


def get_data_loader() -> data_.DataLoader:
    return data_.DataLoader(dataset=DataSet(), batch_size=2, shuffle=True)
