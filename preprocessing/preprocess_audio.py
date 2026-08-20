"""
Audio preprocessing pipeline for Voice-Driven Emotion Separation.

Operations:
1. Load raw audio files
2. Resample audio to 16 kHz
3. Convert stereo audio to mono
4. Normalize waveform amplitude
5. Extract wav2vec 2.0 audio representations
6. Save extracted features

The preprocessing follows the settings described in the paper:
- Sampling rate: 16 kHz
- Feature extractor: wav2vec 2.0
- Frozen encoder during training
"""
import os
import json
import argparse
import hashlib
import random
from pathlib import Path


import numpy as np

import torch
import torchaudio

from tqdm import tqdm

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model
)



CONFIG = {
    "sample_rate": 16000,
    "normalize": True,
    "remove_silence": True,
    "min_duration": 0.2,
    "max_duration": 20.0,
    "chunk_length": 10.0
}


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a"
}



def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)




def file_hash(path):

    h = hashlib.md5()

    with open(path,"rb") as f:

        while True:

            data = f.read(8192)

            if not data:
                break

            h.update(data)


    return h.hexdigest()




def create_directory(path):

    if not os.path.exists(path):

        os.makedirs(path)




def collect_audio_files(root):

    files = []


    for path in Path(root).rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in AUDIO_EXTENSIONS
        ):

            files.append(
                str(path)
            )


    return sorted(files)




def load_audio(path):


    waveform, sr = (
        torchaudio.load(path)
    )


    if waveform.numel()==0:

        raise RuntimeError(
            "Empty waveform"
        )


    waveform = waveform.float()


    return waveform, sr




def convert_mono(waveform):


    if waveform.shape[0] == 1:

        return waveform


    waveform = torch.mean(
        waveform,
        dim=0,
        keepdim=True
    )


    return waveform




def resample(
        waveform,
        sr):


    if sr == CONFIG["sample_rate"]:

        return waveform


    waveform = (
        torchaudio
        .functional
        .resample(
            waveform,
            sr,
            CONFIG["sample_rate"]
        )
    )


    return waveform




def trim_silence(
        waveform,
        threshold=0.015):


    energy = torch.abs(
        waveform
    )


    frame_energy = (
        energy
        .mean(
            dim=0
        )
    )


    indices = (
        frame_energy
        >
        threshold
    )


    if indices.sum()>0:

        waveform = waveform[
            :,
            indices
        ]


    return waveform




def amplitude_normalization(
        waveform):


    maximum = torch.max(
        torch.abs(
            waveform
        )
    )


    if maximum > 0:

        waveform = (
            waveform /
            maximum
        )


    return waveform




def duration_seconds(
        waveform):


    return (
        waveform.shape[-1]
        /
        CONFIG["sample_rate"]
    )




def pad_audio(
        waveform,
        length):


    current = waveform.shape[-1]


    if current >= length:

        return waveform[
            :,
            :length
        ]


    padding = (
        length-current
    )


    waveform = torch.nn.functional.pad(
        waveform,
        (
            0,
            padding
        )
    )


    return waveform




def split_audio(
        waveform):


    chunk_size = int(
        CONFIG["chunk_length"]
        *
        CONFIG["sample_rate"]
    )


    length = waveform.shape[-1]


    if length <= chunk_size:

        return [
            waveform
        ]


    chunks=[]


    start=0


    while start < length:


        end = start + chunk_size


        chunk = waveform[
            :,
            start:end
        ]


        if chunk.shape[-1] < chunk_size:

            chunk = pad_audio(
                chunk,
                chunk_size
            )


        chunks.append(
            chunk
        )


        start=end


    return chunks




def preprocess_waveform(
        path):


    waveform,sr = load_audio(
        path
    )


    waveform = convert_mono(
        waveform
    )


    waveform = resample(
        waveform,
        sr
    )


    if CONFIG["remove_silence"]:

        waveform = trim_silence(
            waveform
        )


    if CONFIG["normalize"]:

        waveform = amplitude_normalization(
            waveform
        )


    duration = duration_seconds(
        waveform
    )


    if duration < CONFIG["min_duration"]:

        raise RuntimeError(
            "Audio too short"
        )


    chunks = split_audio(
        waveform
    )


    return chunks




def load_encoder(
        model_name):


    processor = (
        Wav2Vec2Processor
        .from_pretrained(
            model_name
        )
    )


    model = (
        Wav2Vec2Model
        .from_pretrained(
            model_name
        )
    )


    model.to(
        DEVICE
    )


    model.eval()


    for p in model.parameters():

        p.requires_grad=False


    return processor,model




@torch.no_grad()
def extract_features(
        chunks,
        processor,
        model):


    outputs=[]


    for chunk in chunks:


        audio = (
            chunk
            .squeeze(0)
            .numpy()
        )


        inputs = processor(
            audio,
            sampling_rate=
            CONFIG["sample_rate"],
            return_tensors="pt",
            padding=True
        )


        input_values = (
            inputs
            .input_values
            .to(DEVICE)
        )


        result = model(
            input_values
        )


        feature = (
            result
            .last_hidden_state
            .squeeze(0)
            .cpu()
            .numpy()
        )


        outputs.append(
            feature
        )


    return np.concatenate(
        outputs,
        axis=0
    )




def save_output(
        feature,
        path):


    create_directory(
        os.path.dirname(path)
    )


    np.save(
        path,
        feature
    )




def process_dataset(
        input_dir,
        output_dir,
        model_name):


    create_directory(
        output_dir
    )


    processor,model = load_encoder(
        model_name
    )


    files = collect_audio_files(
        input_dir
    )


    metadata=[]


    for audio_file in tqdm(files):


        try:

            chunks = preprocess_waveform(
                audio_file
            )


            feature = extract_features(
                chunks,
                processor,
                model
            )


            uid = file_hash(
                audio_file
            )


            save_file = os.path.join(
                output_dir,
                uid+".npy"
            )


            save_output(
                feature,
                save_file
            )


            metadata.append(
                {
                    "id":uid,
                    "audio":audio_file,
                    "feature":save_file,
                    "frames":
                    int(feature.shape[0]),
                    "dimension":
                    int(feature.shape[1])
                }
            )


        except Exception as e:


            print(
                "Skip:",
                audio_file,
                e
            )



    with open(
        os.path.join(
            output_dir,
            "metadata.json"
        ),
        "w"
    ) as f:


        json.dump(
            metadata,
            f,
            indent=4
        )




if __name__=="__main__":


    seed_everything()


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--audio_dir",
        required=True
    )


    parser.add_argument(
        "--output_dir",
        required=True
    )


    parser.add_argument(
        "--model",
        default=
        "facebook/wav2vec2-base-960h"
    )


    args = parser.parse_args()


    process_dataset(
        args.audio_dir,
        args.output_dir,
        args.model
    )
