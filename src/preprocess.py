
LETTER_LIST = ['<PAD>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
         'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ', '<SOS>', '<EOS>']


def create_dictionaries(letter_list):
    index2letter = dict()
    letter2index = dict()
    for idx, letter in enumerate(letter_list):
        letter2index[letter_list[idx]] = idx
        index2letter[idx] = letter_list[idx]
    return letter2index, index2letter

letter2index, index2letter = create_dictionaries(LETTER_LIST)

def transform_letter_to_index(transcript, asr_data=False):
    '''
    Transforms text input to numerical input by converting each letter 
    to its corresponding index from letter_list

    Args:
        raw_transcripts: Raw text transcripts with the shape of (N, )
    
    Return:
        transcripts: Converted index-format transcripts. This would be a list with a length of N
    ''' 
    letter_to_index_list = []
    for sentence in transcript:
        letters = [letter2index['<SOS>']]
        for word in sentence:
            if asr_data:
                decoded_word = word.decode('utf-8')
            else:
                decoded_word = word
            for char in decoded_word:
                letters.append(letter2index[char])
            letters.append(letter2index[' '])
        letters.pop()
        letters.append(letter2index['<EOS>'])
        letter_to_index_list.append(letters)
    return letter_to_index_list


