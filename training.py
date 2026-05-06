import numpy as np

def train_model(model, optimizer, loss_fn, x, y, epochs, batch_size):
    loss_history = []
    N = x.shape[0]

    for epoch in range(epochs):
        shuffle_order = np.random.permutation(N)
        for i in range(0, N, batch_size):
            batch = shuffle_order[i: i+batch_size]

            preds = model.forward(x[batch])
            loss = loss_fn.forward(preds, y[batch])

            grad = loss_fn.backward()
            grad = model.backward(grad)

            optimizer.step()

        loss_history.append(loss)
        
        if epoch % (epochs // 10) == 0:
            print(f"epoch={epoch}, loss={loss_history[-1]}")

    return loss_history

        