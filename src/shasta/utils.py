import numpy as  np

def compute_total_accuracy(model, dataloader):
    correct = []

    for xb, yb in dataloader:
        out = model.forward(xb)
        correct.append(np.argmax(out, axis=1) == np.argmax(yb, axis=1))

    correct = np.concatenate(correct)
    return np.mean(correct)