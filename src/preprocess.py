from pydub import AudioSegment
import os
from setup import *

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

def transform_letter_to_index(sentence, asr_data=False):
    '''
    Transforms text input to numerical input by converting each letter 
    to its corresponding index from letter_list

    Args:
        raw_transcripts: Raw text transcripts with the shape of (N, )
    
    Return:
        transcripts: Converted index-format transcripts. This would be a list with a length of N
    ''' 
    letters = [letter2index['<SOS>']]
    for word in sentence:
        if asr_data:
            decoded_word = word.decode('utf-8')
        else:
            decoded_word = word
        for char in decoded_word:
            letters.append(letter2index[char])
    letters.append(letter2index['<EOS>'])
    return letters


def preprocess_wav(audio_path, subtitle_lookup_path, save_dir):
    '''
    Split wav file into small wav files

    Args:
        audio_path: path to knnw wav file
        subtitle_lookup_path: path to knnw_en_sub_wav2vec.csv
        save_dir: directory to save all output wav files
    
    Return:
        nothing
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(save_dir + " alreay exists!")
        return

    audio = AudioSegment.from_file(audio_path, format="wav")
    total_duration = len(audio)
    subtitle_lookup = pd.read_csv(subtitle_lookup_path)
    for i in range(len(subtitle_lookup)):
        start_time = subtitle_lookup.iloc[i, 1]
        stop_time = subtitle_lookup.iloc[i, 2]

        audio_item = audio[start_time: stop_time]
        audio_item.export(save_dir + str(subtitle_lookup.iloc[i, 0]) + ".wav", format="wav")


def preprocess_csv_wav2vec(input_path, output_path):
    df =  pd.read_table(input_path, sep = ";", header=0)
    df.to_csv(output_path, index=False)

    
    
    
def knnw_process_string(text):
    text = text.lower()

    null = 'null'
    text = re.sub(r'.*""', null, text)


    text = text.replace('niiiice', 'nice')
    text = text.replace('shyyy', 'shy')
    text = text.replace('wha ', 'what ')
    text = text.replace('t hahahahaha', 'hahahahaha')
    text = text.replace('scuse', 'excuse')
    text = text.replace('yknow', 'know')
    text = text.replace('ms okudera', 'miss okudera')
    text = text.replace('.', '')
    text = text.replace("'", "")
    text = text.replace('?', '')
    text = text.replace('!', '')
    text = text.replace(',', '')
    text = text.replace('-', ' ')
    text = text.replace('"', '')
    text = text.replace("“", '')
    text = text.replace("”", '')
    text = text.replace('...', '')
    text = text.replace('é', 'e')
    text = text.replace('21', 'twenty one')
    text = text.replace('1200', 'twelve hundred')
    text = text.replace('20th', 'twentieth')
    text = text.replace('7:40', 'seven fourty')
    text = text.replace('8:42', 'eight fourty two')
    text = text.replace('1994', 'nineteen ninety four')
    text = text.replace('9', 'nine')
    text = text.replace('500', 'five hundred')
    text = re.sub(r'\(.*\)', '', text)
    text = re.sub(r'[\w ]+: ', ' ', text)
    text = re.sub(r' +', ' ', text)
    if text[0] == ' ':
        text = text[1:]
    text = re.sub(r'\[.*\] *', ' ', text)
    if text == '':
        text = null
    text = text.replace('n n no u uh', 'uh')
    
    text = text.strip()

    return text

