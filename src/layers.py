from setup import *

def pyramid_reshape(unpacked_out, curr_lengths):
    batch_size = unpacked_out.size(0)

    # handle odd seq_len
    unpacked_out = unpacked_out[:, :(unpacked_out.size(1) // 2) * 2, :]

    # reshape for next pyramid layer
    unpacked_out = unpacked_out.reshape(unpacked_out.size(0), unpacked_out.size(1) // 2, unpacked_out.size(2) * 2)
    curr_lengths = curr_lengths // 2

    return unpacked_out, curr_lengths

class GLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, pyramidal):
        super(GLSTM, self).__init__()
        self.pyramidal = pyramidal
        if pyramidal:
            input_size = 2 * input_size
        
        self.lstm = nn.LSTM(input_size=input_size
                            , hidden_size=hidden_size
                            , num_layers=1
                            , bidirectional=True
                            , batch_first=True
                            , bias=True)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, curr_lengths):
        if self.pyramidal:
            x, curr_lengths = pyramid_reshape(x, curr_lengths)
        packed = pack_padded_sequence(x, curr_lengths, enforce_sorted=False, batch_first=True)
        packed, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(packed, batch_first=True)
        unpacked = self.dropout(unpacked)
        return unpacked, curr_lengths

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.sum(-true_dist * pred, dim=self.dim)