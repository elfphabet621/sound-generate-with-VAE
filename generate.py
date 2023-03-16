from vae import VAE
from sound_data import GuitarChordDataset
from generator import SoundGenerator
from loss_funcs import calculate_combined_loss
import torch
from sound_data import build_dataloader
import numpy 

LEARNING_RATE = 0.00001
BATCH_SIZE = 16
EPOCHS = 100

reconstruction_loss_weight = 10000
audio_dir = 'chord_data'
saved_data_dir = 'saved_data'
target_sample_rate = 22050
duration = 0.74
num_samples = int(44100*duration)
frame_size = 512
hop_length = 256

def select_spectrograms(dataset, num_selected, total_num):
    samples = dict()
    list_idx = numpy.random.randint(0, total_num, size= num_selected)

    samples['feature'] = torch.stack([dataset[i]['feature'] for i in list_idx])
    samples['min'] = torch.stack([dataset[i]['min'] for i in list_idx])
    samples['max'] = torch.stack([dataset[i]['max'] for i in list_idx])
    
    return samples

dataset = GuitarChordDataset(audio_dir, saved_data_dir, 
                        target_sample_rate, 
                        num_samples, frame_size)

vae = VAE(
conv_filters=(64, 32, 16),
conv_kernels=(3, 3, 3),
conv_strides=(2, 2, (2,1)),
latent_space_dim=32
)

generator = SoundGenerator(vae, frame_size)
samples = select_spectrograms(2, len(dataset))
generator.generate(samples, target_sample_rate)