import pandas
import numpy
import torch




print("Subtitle Lookup Preview:")
pandas.read_table("../../datasets/knnw/knnw_en_sub.csv", sep = ";", header=0).head()

print("Audio Shape:")
numpy.load("../../datasets/knnw/knnw_en.spectrogram.npy").shape

class KnnwAudioDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 audio_path,
                 subtitle_lookup_path,
                 total_frames=TOTAL_FRAMES, 
                 total_duration=TOTAL_DURATION):
        
        self.duration_per_frame = total_duration / total_frames
        
        self.audio = numpy.load(audio_path)
        
        self.subtitle_lookup = pandas.read_table(subtitle_lookup_path, 
                                                 sep = ";", header=0)
        
        self.length = len(self.subtitle_lookup)
        
    def __len__(self):
        
        return self.length
    
    def __getitem__(self, i):
        
        start_time = self.subtitle_lookup.iloc[i, 1]
        stop_time = self.subtitle_lookup.iloc[i, 2]
        
        audio_range = self.get_range(start_time, stop_time)
        
        audio_item = self.audio[:,audio_range]
        
        subtitle_item = self.subtitle_lookup.iloc[i, 3]
        subtitle_item = self.get_tokenization(subtitle_item)
        
        return audio_item, subtitle_item
        
    def get_index(self, time, start_flag):
        
        if start_flag == True:
            return numpy.floor(time/self.duration_per_frame)
        
        else:
            return numpy.ceil(time/self.duration_per_frame)
        
    def get_range(self, start_time, end_time):
        
        start_index = self.get_index(start_time, start_flag=True)
        stop_index  = self.get_index(end_time, start_flag=False)
        
        return range(int(start_index), int(stop_index))
    
    def get_tokenization(self, subtitle_item):
        
        return subtitle_item

dataset = KnnwAudioDataset()