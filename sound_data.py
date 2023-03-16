import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
import os, pickle

class MinMaxNormalise():
    def __init__(self, min=0, max=1):
        self.max = max
        self.min = min
    
    def normalise(self, array):
        org_min, org_max = array.min(), array.max()
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array, org_min, org_max
    
    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array

class GuitarChordDataset(Dataset):
    def __init__(self, audio_dir, saved_data_dir, target_sample_rate, num_samples, frame_size):    
        self.audio_dir = audio_dir
        self.saved_data_dir = saved_data_dir
        self.audio_list = os.listdir(audio_dir)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.frame_size = frame_size
        
    def __len__(self):
        return len(self.audio_list)
    
    def __getitem__(self, idx):
        audio_at_idx = self.audio_list[idx].split('.')[0] + '.pkl'
        with open(os.path.join(self.saved_data_dir, audio_at_idx), 'rb') as f:
            signal = pickle.load(f)

        return signal
    
    def processingpipeline(self):
        min_max_norm = MinMaxNormalise()
        for idx in self.audio_list:
            signal_path = os.path.join(self.audio_dir, idx)
            signal, sr = torchaudio.load(signal_path)
            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)
            signal = torchaudio.transforms.Spectrogram(n_fft=self.frame_size)(signal)[:, :-1, :]
            signal, org_min, org_max = min_max_norm.normalise(signal)
            
            self.save_dt(os.path.join(self.saved_data_dir, idx.split('.')[0]), 
                        signal, org_min, org_max)


    def save_dt(self, filepath, feature, org_min, org_max):
        dt_name= filepath+'.pkl'
        with open(dt_name, 'wb') as f:
            pickle.dump({'feature': feature,
                         'min': org_min,
                         'max': org_max}, f)
            
        print("Saved:", dt_name)

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        len_signal = signal.shape[1]
        if  len_signal < self.num_samples:
            num_missing_samples = self.num_samples - len_signal
            signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            signal = torchaudio.transforms.Resample(sr, self.target_sample_rate)(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

def build_dataloader(data, batch_size, mode='train'):
    return DataLoader(data, batch_size=batch_size, 
                      shuffle=True if mode == 'train' else False, num_workers=2)