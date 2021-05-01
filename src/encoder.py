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
        prev_layer_type = None

        for i, layer_type in enumerate(arch):
            if layer_type.startswith("conv"):
                if layer_type.endswith("1x1"):
                    next_layer = ConvLayer(in_channels=1
                                        , out_channels=1
                                        , kernel_size=1
                                        , stride=1
                                        , padding=0)
                elif layer_type.endswith("3x3"):
                    next_layer = ConvLayer(in_channels=1
                                        , out_channels=1
                                        , kernel_size=3
                                        , stride=1
                                        , padding=1)
                elif layer_type.endswith("extractor"):
                    next_layer = ConvExtractor(in_channels=1
                                            , out_channels=cfg['conv_channels'][0]
                                            , kernel_size=3
                                            , stride=1
                                            , padding=1
                                            , pool_stride=2
                                            )
                    curr_feat_dim *= cfg['conv_channels'][0]
            elif layer_type == "lstm" or layer_type == "plstm":
                next_layer = GLSTM(input_size=curr_feat_dim
                                , hidden_size=encoder_hidden_dim
                                , dropout_prob=dropout_prob
                                , pyramidal=(layer_type=="plstm")
                                , proj_size=cfg['proj_size']
                                )
                curr_feat_dim = 2 * encoder_hidden_dim
                
                if cfg['proj_size'] > 0:
                    curr_feat_dim = 2 * cfg['proj_size']
            elif layer_type == "tpool":
                next_layer = TimePool(kernel_size=(3, 1), stride=(2, 1), padding=0)
            
            prev_layer_type = layer_type
            layers.append(next_layer)
        
        self.encoder = nn.ModuleList(layers)

        linear_input_dim = 2 * encoder_hidden_dim
        if cfg['proj_size'] > 0:
            linear_input_dim = 2 * cfg['proj_size']
        
        self.key_network = nn.Linear(linear_input_dim, key_value_size, bias=True)
        self.value_network = nn.Linear(linear_input_dim, key_value_size, bias=True)

    def forward(self, x, input_lengths):
        out = x
        curr_lengths = input_lengths

        for i, layer in enumerate(self.encoder):
            # print(f"layer idx {i}, curr out size: {out.size()}, curr_lengths[0]: {curr_lengths[0]}")
            out, curr_lengths = layer(out, curr_lengths)

        key = self.key_network(out)
        value = self.value_network(out)
        return key, value, curr_lengths
