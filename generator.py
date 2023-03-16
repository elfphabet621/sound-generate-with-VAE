from sound_data import MinMaxNormalise
import torchaudio
import torch

class SoundGenerator:
    def __init__(self, vae, frame_size) -> None:
        self.vae = vae
        self.frame_size = frame_size
        self.min_max_norm = MinMaxNormalise()
    
    def generate(self, samples, target_sample_rate):
        self.vae.eval()
        with torch.inference_mode():
            generated_spectrograms, _, _  = self.vae(samples['feature'])
        self.spec_to_audio(generated_spectrograms, samples['min'], samples['max'], target_sample_rate)
    
    def spec_to_audio(self, spectrograms, min_values, max_values, target_sample_rate):
        signals = []
        for spectrogram, min_value, max_value in zip(spectrograms, min_values, max_values):
            denorm_spec = self.min_max_norm.denormalise(spectrogram,
                                                        min_value,
                                                        max_value)
            signal = torchaudio.transforms.GriffinLim(n_fft=self.frame_size-1)(denorm_spec)
            signals.append(signal)

        torchaudio.save(f'generated.wav', torch.cat(signals, dim=1), target_sample_rate)
