import numpy as np

class TensorDataset:
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        N = len(self.dataset)

        if self.shuffle:
            self.order = np.random.permutation(N)
        else:
            self.order = np.arange(N)

        self.idx = 0
        return self
    
    def __next__(self):
        N = len(self.dataset)

        if self.idx >= N:
            raise StopIteration
        
        batch_idx = self.order[self.idx: self.idx + self.batch_size]

        if self.drop_last and len(batch_idx) < self.batch_size:
            raise StopIteration
        
        self.idx += self.batch_size

        X_batch, y_batch = self.dataset[batch_idx]
        return X_batch, y_batch
    
    def __len__(self):
        N = len(self.dataset)
        if self.drop_last:
            return N // self.batch_size
        else:
            return np.int(np.ceil(N // self.batch_size))
