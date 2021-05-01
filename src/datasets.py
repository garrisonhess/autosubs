from setup import *
from preprocess import *

def spec_augment(X, max_freq_mask=14, max_time_mask=120, nfreq_masks=1, ntime_masks=1, max_time_pct=0.2):

    time_steps = X.shape[0]
    freq_dim = X.shape[1]
    time_mask_cap = min(max_time_pct * time_steps, max_time_mask)

    # frequency masking
    for i in range(nfreq_masks):
        mask_dim = np.random.randint(low=0, high=max_freq_mask)
        mask_start = random.randint(0, freq_dim - mask_dim)
        X[:, mask_start:mask_start + mask_dim] = 0

    # time masking
    for i in range(ntime_masks):
        mask_dim = np.random.randint(low=0, high=time_mask_cap)
        mask_start = random.randint(0, time_steps - mask_dim)
        X[mask_start:mask_start + mask_dim, :] = 0

    return X


class ASRDataset(Dataset):
    def __init__(self, X_path, Y_path="", transforms=None):
        self.X = np.load(X_path, allow_pickle=True, encoding='bytes')
        self.transforms=transforms
        print(f"transforms: {self.transforms}")

        # sort the y's correspondingly
        self.Y = None
        if Y_path != "":
            raw_train_transcript = np.load(Y_path, allow_pickle=True, encoding='bytes')
            self.Y = np.array(transform_letter_to_index(raw_train_transcript, debug=False), dtype=object)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X = self.X[index]
        if self.transforms is not None:
            X = self.transforms(X)
        
        inputs = torch.tensor(X, dtype=torch.float32)
        input_lengths = inputs.shape[0]

        if self.Y is None:
            return inputs, input_lengths

        targets = torch.tensor(self.Y[index], dtype=torch.long)
        target_lengths = targets.shape[0]
        return inputs, input_lengths, targets, target_lengths


class ASRTestDataset(Dataset):
    def __init__(self, X_path):
        self.X = np.load(X_path, allow_pickle=True, encoding='bytes')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        inputs = torch.tensor(self.X[index], dtype=torch.float32)
        input_lengths = inputs.shape[0]
        return inputs, input_lengths


class ToyDataset(Dataset):
    def __init__(self, X_path, Y_path=""):
        self.X = np.load(X_path, allow_pickle=True, encoding='bytes')
        self.Y = None

        if Y_path == "":
            return
        
        # sort all the X's by sequence length ascending
        x_lengths = [self.X[i].shape[0] for i in range(self.X.shape[0])]
        sorted_idxs = np.argsort(x_lengths, axis=-1, kind=None, order=None)
        self.X = self.X[sorted_idxs]

        # sort the y's correspondingly
        if Y_path != "":
            raw_train_transcript = np.load(Y_path, allow_pickle=True, encoding='bytes')
            self.Y = np.array(transform_letter_to_index(raw_train_transcript, debug=True), dtype=object)
            self.Y = self.Y[sorted_idxs]


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        inputs = torch.tensor(self.X[index], dtype=torch.float32)
        input_lengths = inputs.shape[0]

        if self.Y is None:
            return inputs, input_lengths

        targets = torch.tensor(self.Y[index], dtype=torch.long)
        target_lengths = targets.shape[0]
        return inputs, input_lengths, targets, target_lengths


def collate(batch):
    """
    Return:
        pad_x: the padded x (training/validation speech data) 
        pad_y: the padded y (text labels - transcripts)
        x_len: the length of x
        y_len: the length of y
    """
    inputs, input_lengths, targets, target_lengths = list(zip(*batch))
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    inputs = list(inputs)
    targets = [t[1:] for t in targets] # shift target forward by one for <sos> and <eos> haandling
    target_lengths = target_lengths - 1

    # pad the input sequences (batch_size, features, time)
    inputs = pad_sequence(inputs, batch_first=True)
    
    # pad the targets
    targets = pad_sequence(targets, batch_first=True)

    return inputs, targets, input_lengths, target_lengths


def collate_test(batch):
    """
    Return:
        pad_x: the padded x (testing speech data) 
        x_len: the length of x
    """
    inputs, input_lengths = list(zip(*batch))
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    inputs = list(inputs)

    # pad the input sequences (batch_size, features, time)
    inputs = pad_sequence(inputs, batch_first=True)

    return inputs, input_lengths

import pandas
import numpy
import torch

class KnnwAudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 audio_path=knnw_audio_path,
                 subtitle_lookup_path=knnw_subtitle_path,
                 total_frames=1370582, 
                 total_duration=6396010):
        
        self.duration_per_frame = total_duration / total_frames
        
        self.audio = numpy.load(audio_path)
        
        self.subtitle_lookup = pandas.read_table(subtitle_lookup_path, 
                                                 sep = ";", header=0)
        
        self.length = len(self.subtitle_lookup)
        
    def __len__(self):
        
        return self.length
    
    def __getitem__(self, i):
        """getitem for Knnw

        Args:
            i (int): index

        Returns:
            [type]: [description]
        """
        
        start_time = self.subtitle_lookup.iloc[i, 1]
        stop_time = self.subtitle_lookup.iloc[i, 2]
        
        audio_range = self.get_range(start_time, stop_time)
        
        audio_item = self.audio[:,audio_range]
        audio_item_length = int(audio_item.shape[0])
        
        subtitle_item = self.subtitle_lookup.iloc[i, 3]
        subtitle_item = self.get_tokenization(subtitle_item)

        subtitle_item = self.remove_chars(subtitle_item)
        subtitle_item = np.array(transform_letter_to_index([subtitle_item]))[0]
        
        return audio_item, subtitle_item, audio_item_length
        
    def get_index(self, time, start_flag):
        """gets index from timestamp

        Args:
            time (number): timestamp
            start_flag (boolean): floor or ceil rounding

        Returns:
            [type]: [description]
        """
        if start_flag == True:
            return numpy.floor(time/self.duration_per_frame)
        
        else:
            return numpy.ceil(time/self.duration_per_frame)
        
    def get_range(self, start_time, end_time):
        """get_range

        Args:
            start_time (number): start time 
            end_time (number)): end time

        Returns:
            [type]: all data items inside the range
        """
        start_index = self.get_index(start_time, start_flag=True)
        stop_index  = self.get_index(end_time, start_flag=False)
        
        return range(int(start_index), int(stop_index))
    
    def get_tokenization(self, subtitle_item):
        
        return subtitle_item

    def remove_chars(self, text):
        text = text.lower()

        null = 'null'
        text = re.sub(r'.*""', null, text)
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
        return text