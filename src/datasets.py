from transformers import  Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import librosa

from setup import *
from preprocess import *


def spec_augment(X, max_freq_mask=57, max_time_mask=80, nfreq_masks=1, ntime_masks=1, max_time_pct=0.2):

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


def pad_collate_fn(batch):
    # data comes in as (batch, seq_len, feature_len)
    audio, audio_lengths, subtitles, subtitle_lengths = list(zip(*batch))
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
    subtitle_lengths = torch.tensor(subtitle_lengths, dtype=torch.long)

    # offset target subtitles by 1 and decr their lengths
    subtitles = [s[1:] for s in subtitles]
    subtitle_lengths = subtitle_lengths - 1

    # pad the inputs and targets
    audio = pad_sequence(audio, batch_first=True)
    subtitles = pad_sequence(subtitles, batch_first=True)
        
    return audio, subtitles, audio_lengths, subtitle_lengths


class KnnwAudioDataset(torch.utils.data.Dataset):

    def __init__(self, audio_path, subtitle_lookup_path, total_frames, total_duration, spec_aug=False, freq=57, time=35, nfreq_masks=1, ntime_masks=1, max_time_pct=0.2):
        self.duration_per_frame = total_duration / total_frames
        self.audio = np.load(audio_path)
        self.subtitle_lookup = pd.read_table(subtitle_lookup_path, sep = ";", header=0)
        self.length = len(self.subtitle_lookup)
        self.spec_aug = spec_aug
        self.freq = freq
        self.time = time
        self.nfreq_masks = nfreq_masks
        self.ntime_masks = ntime_masks
        self.max_time_pct = max_time_pct
    
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
        audio_item = self.audio[:, audio_range].T
        audio_item_length = int(audio_item.shape[0])

        # apply SpecAugment if specified
        if self.spec_aug:
            audio_item = spec_augment(audio_item
                                    , max_freq_mask=self.freq
                                    , max_time_mask=self.time
                                    , nfreq_masks=self.nfreq_masks
                                    , ntime_masks=self.ntime_masks
                                    , max_time_pct=self.max_time_pct)

        # convert audio to tensor
        audio_item = torch.tensor(audio_item, dtype=torch.float32)
        
        # process subtitles
        subtitle_item = self.subtitle_lookup.iloc[i, 3]
        subtitle_item = self.remove_chars(subtitle_item)
        subtitle_item = transform_letter_to_index(subtitle_item)
        subtitle_item = torch.tensor(subtitle_item, dtype=torch.long)

        return audio_item, audio_item_length, subtitle_item, len(subtitle_item)
        
    def get_index(self, time, start_flag):
        """gets index from timestamp
        Args:
            time (number): timestamp
            start_flag (boolean): floor or ceil rounding
        Returns:
            [type]: [description]
        """
        if start_flag == True:
            return np.floor(time/self.duration_per_frame)
        
        return np.ceil(time/self.duration_per_frame)
        
    def get_range(self, start_time, end_time):
        """get_range
        Args:
            start_time (number): start time 
            end_time (number)): end time
        Returns:
            [type]: all data items inside the range
        """
        start_index = self.get_index(start_time, start_flag=True)
        stop_index = self.get_index(end_time, start_flag=False)
        return range(int(start_index), int(stop_index))
    
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



def sub_file_edit():
    sub_file = open(cfg['knnw_subtitle_path'], 'r')
    sub_file_edit = open(cfg['knnw_subtitle_edit_path'], 'w')
    lines = sub_file.readlines()
    sub_file_edit.write(lines[0])

    num = 1
    for line in lines[2:]:
        try:
            line.index('""')
        except ValueError: 
            s = line[line.index(';'):]
            s = str(num) + s
            sub_file_edit.write(s)
        
        num += 1


class ASRDataset(Dataset):
    def __init__(self, X_path, Y_path="", transforms=None):
        self.X = np.load(X_path, allow_pickle=True, encoding='bytes')
        self.transforms=transforms
        print(f"transforms: {self.transforms}")

        # sort the y's correspondingly
        self.Y = None
        if Y_path != "":
            raw_train_transcript = np.load(Y_path, allow_pickle=True, encoding='bytes')
            self.Y = np.array(transform_letter_to_index(raw_train_transcript, asr_data=True), dtype=object)

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
    targets = [t[1:] for t in targets] # shift target forward by one for <sos> and <eos> handling
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


class Wav2vec2ExtractorDataset(Dataset):
    '''dataset that extract features from raw wav audio based on Wav2vec2 pretrained feature extracor'''

    def __init__(self, subtitle_lookup_path, wav_dir, pretrained_chpt="facebook/wav2vec2-base-960h", num_proc=4, preprocess_batch_size=8):
        """
        Args:
            subtitle_lookup_path (NOTE this file must be compatible with pd.read_csv())
            wav_dir: directory of wav files e.g. './wav_data/'
            pretrained_chpt: pretrained checkpoints to load
            num_proc: number of processes allowed when doing dataset preprocessing
            preprocess_batch_size: this is ONLY used inside this dataset to preprocess wav files faster
        Returns:
            
        """
        self.subtitle_lookup_path = subtitle_lookup_path
        self.wav_dir = wav_dir
        self.num_proc = num_proc
        self.preprocess_batch_size = preprocess_batch_size

        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_chpt)
        self.feature_extractor = Wav2Vec2ForCTC.from_pretrained(pretrained_chpt).wav2vec2.feature_extractor

        self.knnw_prepared = self.setup_dataset()

    def __len__(self):
        return len(self.subtitle_lookup)

    def __getitem__(self, index):
        x = self.processor(
            self.knnw_prepared['data'][index]["input_values"], 
            sampling_rate=16000, 
            return_tensors="pt").input_values
        
        subtitle_item = torch.tensor(self.knnw_prepared['data'][index]["labels"], dtype=torch.long)
        subtitle_item_length = len(subtitle_item)

        audio_item = self.feature_extractor(x).squeeze().T  # shape (seq_len, 512)
        audio_item_length = audio_item.shape[1]

        return audio_item, audio_item_length, subtitle_item, subtitle_item_length
    
    def setup_dataset(self):
        def remove_chars(batch):
            batch['Text'] = batch['Text'].lower()
            null = 'null'
            batch['Text'] = re.sub(r'.*""', null, batch['Text'])
            batch['Text'] = batch['Text'].replace('?', '')
            batch['Text'] = batch['Text'].replace('!', '')
            batch['Text'] = batch['Text'].replace(',', '')
            batch['Text'] = batch['Text'].replace('-', ' ')
            batch['Text'] = batch['Text'].replace('"', '')
            batch['Text'] = batch['Text'].replace("“", '')
            batch['Text'] = batch['Text'].replace("”", '')
            batch['Text'] = batch['Text'].replace('...', '')
            batch['Text'] = batch['Text'].replace('é', 'e')
            batch['Text'] = batch['Text'].replace('21', 'twenty one')
            batch['Text'] = batch['Text'].replace('1200', 'twelve hundred')
            batch['Text'] = batch['Text'].replace('20th', 'twentieth')
            batch['Text'] = batch['Text'].replace('7:40', 'seven fourty')
            batch['Text'] = batch['Text'].replace('8:42', 'eight fourty two')
            batch['Text'] = batch['Text'].replace('1994', 'nineteen ninety four')
            batch['Text'] = batch['Text'].replace('9', 'nine')
            batch['Text'] = batch['Text'].replace('500', 'five hundred')
            batch['Text'] = re.sub(r'\(.*\)', '', batch['Text'])
            batch['Text'] = re.sub(r'[\w ]+: ', ' ', batch['Text'])
            batch['Text'] = re.sub(r' +', ' ', batch['Text'])
            if batch['Text'][0] == ' ':
                batch['Text'] = batch['Text'][1:]
            batch['Text'] = re.sub(r'\[.*\] *', ' ', batch['Text'])
            if batch['Text'] == '':
                batch['Text'] = null
            
            return batch

        def speech_file_to_array_fn(batch):
            speech_array, sr = librosa.load(self.wav_dir + str(batch['Number']) + ".wav", sr=16000)
            batch["speech"] = speech_array
            batch["sampling_rate"] = sr
            batch["target_text"] = transform_letter_to_index(batch["Text"])
            return batch

        def prepare_dataset(batch):
            # check that all files have the correct sampling rate
            assert (
                len(set(batch["sampling_rate"])) == 1
            ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

            batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
            with processor.as_target_processor():
                batch["labels"] = batch["target_text"]

            return batch

        knnw = load_dataset('csv', data_files={'data': self.subtitle_lookup_path})
        knnw = knnw.map(remove_chars)
        knnw = knnw.map(speech_file_to_array_fn, remove_columns=knnw.column_names["data"], num_proc=self.num_proc)
        knnw = knnw.map(prepare_dataset,
                        remove_columns=knnw.column_names["data"],
                        batch_size=self.preprocess_batch_size,
                        num_proc=self.num_proc,
                        batched=True)

        return knnw
        