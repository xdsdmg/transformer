import torch.nn as nn
import torch.optim as optim
import data
import model
import torch

EPOCH_TOTAL = 1000
MODEL_SAVE_PATH = "MyTransformer.pth"


def train():
    m = model.Transformer(data.SRC_VOCAB_SIZE, data.TGT_VOCAB_SIZE)
    m.train()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(m.parameters(), lr=1e-3, momentum=0.99)

    loader = data.get_data_loader()

    for epoch in range(EPOCH_TOTAL):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            output = m(enc_inputs, dec_inputs)
            loss = criterion(output, dec_outputs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/1000], Loss: {loss.item()}")

    torch.save(m, MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()
