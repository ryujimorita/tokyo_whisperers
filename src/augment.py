from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    PitchShift,
    TimeStretch,
)


class DataAugmentator:
    def __init__(self, audio_column_name: str):
        self.audio_column_name = audio_column_name

        # init Compose class
        self.augmentation = Compose(
            [
                TimeStretch(
                    min_rate=0.9,
                    max_rate=1.1,
                    p=0.2,
                    leave_length_unchanged=False,
                ),
                Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                AddGaussianNoise(
                    min_amplitude=0.005, max_amplitude=0.015, p=1.0
                ),
            ]
        )

    def augment_dataset(self, batch):
        # load and (possibly) resample audio data to 16kHz
        sample = batch[self.audio_column_name]

        # apply augmentation
        augmented_waveform = self.augmentation(
            sample["array"], sample_rate=sample["sampling_rate"]
        )
        sample["array"] = augmented_waveform
        return batch
