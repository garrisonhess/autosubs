from setup import *
from encoder import Encoder
from decoder import Decoder

class Seq2Seq(nn.Module):
    '''
    Seq2Seq wraps the encoder and decoder together
    '''
    def __init__(self, input_dim
                , vocab_size
                , encoder_hidden_dim
                , decoder_hidden_dim
                , embed_dim
                , key_value_size
                , enc_dropout
                , dec_dropout
                , encoder_arch
                , use_multihead
                , nheads
                ):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim=input_dim
                            , encoder_hidden_dim=encoder_hidden_dim
                            , key_value_size=key_value_size
                            , dropout_prob=enc_dropout
                            , arch=encoder_arch
                            )
        
        self.decoder = Decoder(vocab_size=vocab_size
                            , decoder_hidden_dim=decoder_hidden_dim
                            , embed_dim=embed_dim
                            , key_value_size=key_value_size
                            , dec_dropout_prob=dec_dropout
                            , use_multihead=use_multihead
                            , nheads=nheads
                            )

    def forward(self, inputs, input_lengths, device, teacher_forcing, targets=None, mode='train', attn_forcing=0.0):
        key, value, encoded_seq_lens = self.encoder(inputs, input_lengths)
        predictions, attention = self.decoder(key=key
                                            , value=key
                                            , encoded_seq_lens=encoded_seq_lens
                                            , teacher_forcing=teacher_forcing
                                            , device=device
                                            , targets=targets
                                            , mode=mode)
        return predictions, attention, encoded_seq_lens
