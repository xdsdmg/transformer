import data
import torch
import train


def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    next_symbol = start_symbol
    flag = True
    while flag:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype)], -1)
        dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        projected = projected.squeeze(0)
        max_ = projected.max(dim=-1, keepdim=False)
        prob = max_[1]
        next_symbol = prob.data[-1]
        if next_symbol == data.TGT_VOCAB['.']:
            flag = False
        print(next_symbol)
    return dec_input


if __name__ == "__main__":
    model = torch.load(train.MODEL_SAVE_PATH)
    model.eval()
    loader = data.get_data_loader()
    with torch.no_grad():
        enc_inputs, _, _ = next(iter(loader))
        for i in range(len(enc_inputs)):
            greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=data.TGT_VOCAB['S'])
            predict = model(enc_inputs[i].view(1, -1), greedy_dec_input)
            predict = predict.data.max(dim=-1, keepdim=False)[1]
            print(enc_inputs[i], '->', [data.IDX_TO_WORD[n.item()] for n in predict])
