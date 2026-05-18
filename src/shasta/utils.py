import numpy as  np

def batch_total_accuracy(model, x, labels, batch_size=8):
    predictions = []
    N = x.shape[0]

    for i in range(0, N, batch_size):
        batch = x[i: i+batch_size]
        out = model.forward(batch)
        predictions.append(np.argmax(out, axis=1))

    predictions = np.concatenate(predictions)
    return np.mean(predictions == labels)