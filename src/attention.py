from setup import *

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask):
        scaling_factor = query.size(-1)
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2) / math.sqrt(scaling_factor)

        # apply mask here (with negative infinity)
        energy.masked_fill_(mask, -1e16)
        attention = nn.functional.softmax(energy, dim=1)

        # true attention to concatenate with previous hidden state
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)

        return context, attention.clone().detach().cpu().numpy()[0]


