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
        subtitle_item = knnw_process_string(subtitle_item)
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
