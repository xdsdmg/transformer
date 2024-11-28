import torch

import torch.utils.data as data_


class DataSet(data_.Dataset):
    def __init__(self, encoder_inputs: torch.LongTensor, decoder_inputs: torch.LongTensor, decoder_outputs: torch.LongTensor):
        super(DataSet, self).__init__()
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_outputs = decoder_outputs

    def __len__(self) -> int:
        return self.encoder_inputs.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder_inputs[idx], self.decoder_inputs[idx], self.decoder_outputs[idx]
