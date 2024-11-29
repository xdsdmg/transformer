import torch.nn as nn
import torch.optim as optim
import data
import model

EPOCH_TOTAL = 1


def train():
    m = model.Transformer(data.SRC_VOCAB_SIZE, data.TGT_VOCAB_SIZE)
    m.train()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(m.parameters(), lr=1e-3, momentum=0.99)

    data_set = data.DataSet()
    for epoch in range(EPOCH_TOTAL):
        for enc_input, dec_input, dec_output in data_set:
            print(enc_input, dec_input, dec_output)

            output = m(enc_input, dec_input)
            loss = criterion(output, dec_output.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/1000], Loss: {loss.item()}")


if __name__ == "__main__":
    train()
