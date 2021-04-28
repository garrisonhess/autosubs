from setup import *

class ConvLSTM(nn.Module):
    def __init__(self
                , in_channels
                , out_channels
                , kernel_size
                , stride
                , feat_dim
                , lstm_input_dim
                , lstm_hidden_dim
                , layer_id
                , pool_stride):
        super().__init__()

        self.padding_dim = (kernel_size[0]//2, kernel_size[1]//2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=self.padding_dim, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(9, 3), stride=self.pool_stride, padding=1)
        self.lstm1 = nn.LSTM(input_size=lstm_input_dim
                            , hidden_size=lstm_hidden_dim
                            , num_layers=1
                            , bidirectional=True
                            , batch_first=True
                            , bias=True)
        

    def forward(self, x, input_lengths):
        # data comes into a convblock as (batch_size, seq_len, feat_dim)
        # add conv dim for (batch_size, in_channels, feat_dim, seq_len)
        batch_size, seq_len, feat_dim = x.size()


        # lstm to conv requires this
        x = x.transpose(1, 2).unsqueeze(1)
        x = x.view(batch_size, self.in_channels, self.feat_dim, seq_len)



        # print(f"in size {x.size()}")
        # print(f"inn legnths: {input_lengths}")
        curr_lengths = torch.tensor([z // self.time_downsample for z in input_lengths], dtype=torch.long)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.pool(out)



        # remove channel dim (batch_size, out_channels * feat_dim, seq_len) -> (batch_size, seq_len, features)
        # conv to lstm requires this
        out = out.transpose(1, 3).flatten(2)

        # print(f"out size {out.size()}")
        # print(curr_lengths)
        out = pack_padded_sequence(out, curr_lengths, enforce_sorted=False, batch_first=True)
        out, _ = self.lstm1(out)
        out, curr_lengths = pad_packed_sequence(out, batch_first=True)

        # print(f"lstm out {out.size()}")
        # data out as (batch_size, time, features)
        return out, curr_lengths



