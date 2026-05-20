import numpy as np

def train_model(model, optimizer, loss_fn, TrainLoader, ValidationLoader=None, epochs=5):
    mean_train_loss = []
    validation_loss = []
    cadence = max(epochs // 10, 1)

    for epoch in range(epochs):
        epoch_losses = []

        model.train()
        for xb, yb in TrainLoader:
            preds = model.forward(xb)
            loss = loss_fn.forward(preds, yb)

            grad = loss_fn.backward()
            grad = model.backward(grad)

            optimizer.step()

            epoch_losses.append(loss)

        mean_train_loss.append(np.mean(epoch_losses))

        if ValidationLoader is not None:
            val_losses = []
            for xvb, yvb in ValidationLoader:
                preds = model.forward(xvb)
                loss = loss_fn.forward(preds, yvb)
                val_losses.append(loss)
            validation_loss.append(np.mean(val_losses))
        
        if epoch % cadence == 0:
            print(f"epoch={epoch}, train loss={mean_train_loss[-1]}, validation loss={validation_loss[-1]}")

    return mean_train_loss, validation_loss

        