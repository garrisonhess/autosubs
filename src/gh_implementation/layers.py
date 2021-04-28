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
    '''
    Pyramidal BiLSTM
    '''
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



class ConvLayer(nn.Module):
    def __init__(self
                , in_channels
                , out_channels
                , kernel_size
                , stride
                , padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    # always outputs unpacked in (batch_size, seq_len, out_channels * feat_dim)
    def forward(self, x, input_lengths):
        # data comes into a convblock unpacked as (batch_size, seq_len, feat_dim)
        # add conv dim for (batch_size, in_channels, feat_dim, seq_len)
        x = x.unsqueeze(1)
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = out.squeeze()
        return out, input_lengths


class TimePool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=(1, 0))

    # always outputs unpacked in (batch_size, seq_len, out_channels * feat_dim)
    def forward(self, x, curr_lengths):
        # data comes into a convblock unpacked as (batch_size, seq_len, feat_dim)
        # add conv dim for (batch_size, in_channels, feat_dim, seq_len)
        x = x.unsqueeze(1)
        out = self.pool(x)
        out = out.squeeze(1)
        curr_lengths = curr_lengths // 2
        return out, curr_lengths




class ConvExtractor(nn.Module):
    def __init__(self
                , in_channels
                , out_channels
                , kernel_size
                , stride
                , padding
                , pool_stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

    # always outputs unpacked in (batch_size, seq_len, out_channels * feat_dim)
    def forward(self, x, input_lengths):
        curr_lengths = input_lengths
        # print(f"IN SIZE{x.size()}")
        # print(f"curr_lengths {curr_lengths}")
        # data comes into a convblock unpacked as (batch_size, seq_len, 40)
        # comes out as (batch_size, seq_len, 40 * out_channels)
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        # print(f"PRE pool 1 shape {out.size()}")
        out = self.pool1(out)
        # print(f"after pool 1 shape {out.size()}")
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        # print(f"PRE pool 2 shape {out.size()}")
        out = self.pool2(out)
        # print(f"after pool 2 shape {out.size()}")
        out = out.transpose(1, 2).flatten(2)
        # print(f"FINAL SIZE: {out.size()}")
        curr_lengths = curr_lengths // 4
        # print(f"FINAL curr_lengths: {curr_lengths}")
        return out, curr_lengths





class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_factor=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_factor, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_factor, in_channels, bias=False),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, channels, dim1, dim2 = x.size()
        initial_input = x
        out = self.pool(x).view(batch_size, channels)
        out = self.fc(out).view(batch_size, channels, 1, 1)
        return initial_input * out.expand(initial_input.size())

