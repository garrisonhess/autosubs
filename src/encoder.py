from setup import *
from layers import *

class Encoder(nn.Module):
    def __init__(self
                , input_dim
                , encoder_hidden_dim
                , key_value_size
                , dropout_prob
                , arch):
        super(Encoder, self).__init__()
        layers = []
        curr_feat_dim = cfg["input_dim"]

        for i, layer_type in enumerate(arch):
            next_layer = GLSTM(input_size=curr_feat_dim
                            , hidden_size=encoder_hidden_dim
                            , dropout_prob=dropout_prob
                            , pyramidal=(layer_type=="plstm")
                            )
            curr_feat_dim = 2 * encoder_hidden_dim
            layers.append(next_layer)
        
        self.encoder = nn.ModuleList(layers)
        self.key_network = nn.Linear(curr_feat_dim, key_value_size, bias=True)
        self.value_network = nn.Linear(curr_feat_dim, key_value_size, bias=True)

    def forward(self, x, input_lengths):
        out = x
        curr_lengths = input_lengths

        for i, layer in enumerate(self.encoder):
            # print(f"layer idx {i}, curr out size: {out.size()}, curr_lengths[0]: {curr_lengths[0]}")
            out, curr_lengths = layer(out, curr_lengths)

        key = self.key_network(out)
        value = self.value_network(out)
        return key, value, curr_lengths
