from setup import *
from attention import Attention
from preprocess import *
from layers import *

class Decoder(nn.Module):
    def __init__(self
                , vocab_size
                , decoder_hidden_dim
                , embed_dim
                , key_value_size
                , dec_dropout_prob
                , use_multihead
                , nheads):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.key_value_size = key_value_size
        self.lstm1 = nn.LSTMCell(input_size=embed_dim + key_value_size, hidden_size=decoder_hidden_dim, bias=True)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size, bias=True)
        self.use_multihead = use_multihead
        if self.use_multihead:
            self.nheads = nheads
            self.attention = MultiheadAttention(embed_dim=key_value_size
                                            , num_heads=self.nheads
                                            , dropout=dec_dropout_prob
                                            , bias=True
                                            , add_bias_kv=False
                                            , add_zero_attn=False
                                            , kdim=key_value_size
                                            , vdim=key_value_size)
        else:
            self.attention = Attention()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(2 * key_value_size, vocab_size, bias=True)
        self.embedding.weight = self.fc.weight
        self.dropout1 = nn.Dropout(p=dec_dropout_prob)
        self.dropout2 = nn.Dropout(p=dec_dropout_prob)


    def forward(self, key, value, encoded_seq_lens, device, teacher_forcing, targets=None, mode='train'):
        '''
        Args:
            key :(B, T, key_value_size) - Output of the Encoder Key projection layer
            value: (B, T, key_value_size) - Output of the Encoder Value projection layer
            targets: (B, text_len) - Batch input of text (in index format) with text_length
            mode: Train or eval mode
        Return:
            predictions: the character prediction probability 
        '''
        batch_size, max_encoded_seq_len, attn_kv_size = key.shape

        if mode == 'train' or mode == 'warmup':
            max_target_len = targets.size(1)
            embeddings = self.embedding(targets)
        else:
            max_target_len = 130
        
        # prepare variables for decoding loop
        mask = torch.arange(max_encoded_seq_len).unsqueeze(0).to(device) >= encoded_seq_lens.unsqueeze(1).to(device, non_blocking=True)
        all_preds = []
        curr_preds = torch.zeros(batch_size, 1).to(device, non_blocking=True)
        attentions = np.zeros((max_encoded_seq_len, max_target_len))
        if mode == 'warmup':
            context = torch.zeros((batch_size, self.key_value_size), device=device)
        else:
            context = value[:, 0, :]

        lstm1_hidden = None
        lstm1_cell = None
        lstm2_hidden = None
        lstm2_cell = None

        for seq_idx in range(max_target_len):
            if (mode == 'train' or mode == 'warmup') and torch.rand(1) < teacher_forcing:
                if seq_idx == 0:
                    init_char = torch.zeros(batch_size, dtype=torch.long, device=device).fill_(letter2index['<SOS>'])
                    char_embed = self.embedding(init_char)
                else:
                    # print(f"tf target: {index2letter[targets[:, seq_idx - 1][0].item()]}, model pred: {index2letter[curr_preds.argmax(dim=-1)[0].item()]}")
                    char_embed = embeddings[:, seq_idx - 1, :]
            else:
                char_idx = torch.zeros(batch_size, dtype=torch.long, device=device).fill_(letter2index['<SOS>']) if seq_idx == 0 else curr_preds.argmax(dim=-1)
                char_embed = self.embedding(char_idx)
            
            y_context = torch.cat([char_embed, context], dim=1)

            # inputs: input, (h_0, c_0)
            if lstm1_hidden is not None:
                lstm1_hidden, lstm1_cell = self.lstm1(y_context, (lstm1_hidden, lstm1_cell))
                lstm1_hidden = self.dropout1(lstm1_hidden)
                lstm2_hidden, lstm2_cell = self.lstm2(lstm1_hidden, (lstm2_hidden, lstm2_cell))
            else:
                lstm1_hidden, lstm1_cell = self.lstm1(y_context)
                lstm1_hidden = self.dropout1(lstm1_hidden)
                lstm2_hidden, lstm2_cell = self.lstm2(lstm1_hidden)

            lstm2_hidden = self.dropout2(lstm2_hidden)

            # multihead attention
            # query must be (target_len, batch_size, embedding_dim)
            # key must be (seq_len, batch_size, embedding_dim)
            # value must be (seq_len, batch_size, embedding_dim)
            # key_padding_mask must be (batch_size, seq_len)
            # print(f"query size {lstm2_hidden.size()}")
            # print(f"key size {key.size()}")
            # print(f"value size {value.size()}")
            # print(f"mask size {mask.size()}")
            # Get context from the output of the second LSTM Cell (attention vector size (encoded_seq_lens))
            if mode != 'warmup':
                if self.use_multihead:
                    context, _ = self.attention(query=lstm2_hidden.unsqueeze(0)
                                                , key=key.transpose(0, 1)
                                                , value=value.transpose(0, 1)
                                                , key_padding_mask=mask
                                                , need_weights=False
                                                , attn_mask=None
                                                )
                    context = context.squeeze(0)
                else:
                    context, attentions[:, seq_idx] = self.attention(query=lstm2_hidden
                                                , key=key
                                                , value=value
                                                , mask=mask)

            final_out = torch.cat([lstm2_hidden, context], dim=1)

            # send LSTM2 output and attention context to final FC layer
            curr_preds = self.fc(final_out)
            all_preds.append(curr_preds.unsqueeze(1))

        all_preds = torch.cat(all_preds, dim=1)
        return all_preds, attentions

