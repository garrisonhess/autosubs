from setup import *
from preprocess import *

def spec_augment(X, max_freq_mask=14, max_time_mask=120, nfreq_masks=2, ntime_masks=2):

    time_steps = X.shape[0]
    freq_dim = X.shape[1]

    # frequency masking
    for i in range(nfreq_masks):
        mask_dim = np.random.randint(low=0, high=max_freq_mask)
        mask_start = random.randint(0, freq_dim - mask_dim)
        X[:, mask_start:mask_start + mask_dim] = 0

    # time masking
    for i in range(ntime_masks):
        mask_dim = np.random.randint(low=0, high=max_time_mask)
        mask_start = random.randint(0, time_steps - mask_dim)
        X[mask_start:mask_start + mask_dim, :] = 0

    return X


class ASRDataset(Dataset):
    def __init__(self, X_path, Y_path="", transforms=None):
        self.X = np.load(X_path, allow_pickle=True, encoding='bytes')
        self.transforms=transforms

        # sort all the X's by sequence length ascending
        x_lengths = [self.X[i].shape[0] for i in range(self.X.shape[0])]
        sorted_idxs = np.argsort(x_lengths, axis=-1, kind=None, order=None)
        self.X = self.X[sorted_idxs]

        # sort the y's correspondingly
        self.Y = None
        if Y_path != "":
            raw_train_transcript = np.load(Y_path, allow_pickle=True, encoding='bytes')
            self.Y = np.array(transform_letter_to_index(raw_train_transcript, debug=False), dtype=object)
            self.Y = self.Y[sorted_idxs]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X = self.X[index]
        if self.transforms is not None:
            X = self.transforms(X)
        
        inputs = torch.tensor(X, dtype=torch.float32)
        input_lengths = inputs.shape[0]

        if self.Y is None:
            return inputs, input_lengths

        targets = torch.tensor(self.Y[index], dtype=torch.long)
        target_lengths = targets.shape[0]
        return inputs, input_lengths, targets, target_lengths


class ASRTestDataset(Dataset):
    def __init__(self, X_path):
        self.X = np.load(X_path, allow_pickle=True, encoding='bytes')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        inputs = torch.tensor(self.X[index], dtype=torch.float32)
        input_lengths = inputs.shape[0]
        return inputs, input_lengths


class ToyDataset(Dataset):
    def __init__(self, X_path, Y_path=""):
        self.X = np.load(X_path, allow_pickle=True, encoding='bytes')
        self.Y = None

        if Y_path == "":
            return
        
        # sort all the X's by sequence length ascending
        x_lengths = [self.X[i].shape[0] for i in range(self.X.shape[0])]
        sorted_idxs = np.argsort(x_lengths, axis=-1, kind=None, order=None)
        self.X = self.X[sorted_idxs]

        # sort the y's correspondingly
        if Y_path != "":
            raw_train_transcript = np.load(Y_path, allow_pickle=True, encoding='bytes')
            self.Y = np.array(transform_letter_to_index(raw_train_transcript, debug=True), dtype=object)
            self.Y = self.Y[sorted_idxs]


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        inputs = torch.tensor(self.X[index], dtype=torch.float32)
        input_lengths = inputs.shape[0]

        if self.Y is None:
            return inputs, input_lengths

        targets = torch.tensor(self.Y[index], dtype=torch.long)
        target_lengths = targets.shape[0]
        return inputs, input_lengths, targets, target_lengths


def collate(batch):
    """
    Return:
        pad_x: the padded x (training/validation speech data) 
        pad_y: the padded y (text labels - transcripts)
        x_len: the length of x
        y_len: the length of y
    """
    inputs, input_lengths, targets, target_lengths = list(zip(*batch))
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    inputs = list(inputs)
    targets = [t[1:] for t in targets] # shift target forward by one for <sos> and <eos> haandling
    target_lengths = target_lengths - 1

    # pad the input sequences (batch_size, features, time)
    inputs = pad_sequence(inputs, batch_first=True)
    
    # pad the targets
    targets = pad_sequence(targets, batch_first=True)

    return inputs, targets, input_lengths, target_lengths


def collate_test(batch):
    """
    Return:
        pad_x: the padded x (testing speech data) 
        x_len: the length of x
    """
    inputs, input_lengths = list(zip(*batch))
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    inputs = list(inputs)

    # pad the input sequences (batch_size, features, time)
    inputs = pad_sequence(inputs, batch_first=True)

    return inputs, input_lengths