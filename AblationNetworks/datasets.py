import torch.utils.data as Data


class MyDataSet_freq(Data.Dataset):
    def __init__(self, inputs):
        super(MyDataSet_freq, self).__init__()
        self.inputs = inputs

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx]


class MyDataSet_label(Data.Dataset):
    def __init__(self, inputs, labels):
        super(MyDataSet_label, self).__init__()
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class MyDataSet_freqId(Data.Dataset):
    def __init__(self, inputs, ids):
        super(MyDataSet_freqId, self).__init__()
        self.inputs = inputs
        self.ids = ids

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.ids[idx]


class MyDataSet_id(Data.Dataset):
    def __init__(self, inputs, labels, ids):
        super(MyDataSet_id, self).__init__()
        self.inputs = inputs
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.ids[idx]
