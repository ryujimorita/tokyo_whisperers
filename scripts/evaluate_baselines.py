# %%
# Install Whisper
!pip install git+https://github.com/openai/whisper.git

# %%
# Install system dependencies (libsndfile1 and ffmpeg)
!apt-get install -y libsndfile1 ffmpeg

# %%
# Clone the ReazonSpeech repository
!git clone https://github.com/reazon-research/reazonspeech

# %%
# Install ReazonSpeech Nemo
!pip install --no-warn-conflicts reazonspeech/pkg/nemo-asr

# Install ReazonSpeech K2
!pip install --no-warn-conflicts reazonspeech/pkg/k2-asr

# Install ReazonSpeech ESPnet
!pip install --no-warn-conflicts reazonspeech/pkg/espnet-asr

# %%
!pip install jiwer
!apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
!pip install mecab-python3

# %%
import os

# Directory structure for the audio files:
# - The "basic5000" folder contains the dataset for testing.
# - Inside "basic5000":
#     - A subfolder named "wav" stores multiple audio files (*.wav).
#     - A text file named "transcript_utf8.txt" serves as the ground truth transcription file
#       and contains the corresponding text for the audio files in UTF-8 encoding.
#
# This structure is identical to the "basic5000" folder inside the "jsut_ver1.1" dataset,
# which can be downloaded from the following URL:
# https://sites.google.com/site/shinnosuketakamichi/publication/jsut
#
# The organization ensures compatibility with the JSUT corpus for ASR tasks.


base_dir = "basic5000"
wav_dir = os.path.join(base_dir, "wav")
os.makedirs(wav_dir, exist_ok=True)

# Please manually upload wav files and transcript_utf8.txt from basic5000.

# %%
from reazonspeech.nemo.asr import transcribe as nemo_transcribe, audio_from_path as nemo_audio_from_path, load_model as load_reazon_nemo
from reazonspeech.k2.asr import transcribe as k2_transcribe, audio_from_path as k2_audio_from_path, load_model as load_reazon_k2
from reazonspeech.espnet.asr import transcribe as espnet_transcribe, audio_from_path as espnet_audio_from_path, load_model as load_reazon_espnet
from jiwer import cer, wer
import os
import re
import MeCab
import pandas as pd
from google.colab import files
import whisper
import torchaudio

# Paths
audio_folder_path = "basic5000/wav"
transcript_file_path = "basic5000/transcript_utf8.txt"

# Initialize MeCab
mecab = MeCab.Tagger("-Owakati -r /etc/mecabrc")  # Use wakati mode to tokenize into words

# Normalize text by converting full-width characters to half-width and removing punctuation/spaces
def normalize(text):
    """
    Normalize text by:
    1. Converting full-width characters to half-width.
    2. Removing Japanese punctuation marks (、。) and spaces.
    """
    FULLWIDTH_TO_HALFWIDTH = str.maketrans(
        "　０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ！゛＃＄％＆（）＊＋ー／：；〈＝〉？＠［］＾＿'｛｜｝～",
        ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&()*+-/:;<=>?@[]^_`{|}~'
    )
    # Convert full-width to half-width
    text = text.translate(FULLWIDTH_TO_HALFWIDTH)
    # Remove punctuation and spaces
    text = re.sub(r"[、。 ]", "", text)
    return text

# Function to tokenize text using MeCab
def tokenize_text(text):
    return mecab.parse(text).strip()  # Tokenize and remove trailing newline

# Load ground truth transcripts into a dictionary
def load_transcripts(transcript_file_path):
    transcripts = {}
    with open(transcript_file_path, "r", encoding="utf-8") as f:
        for line in f:
            key, text = line.strip().split(":", 1)
            transcripts[key] = text.strip()
    return transcripts

# Transcribe audio using Whisper
def transcribe_whisper(model, audio_path):
    result = model.transcribe(audio_path, fp16=True)
    return result['text'].strip()

# Transcribe audio using ReazonSpeech Nemo
def transcribe_reazon_nemo(model, audio_path):
    audio = nemo_audio_from_path(audio_path)
    result = nemo_transcribe(model, audio)
    return " ".join([seg.text for seg in result.segments])

# Transcribe audio using ReazonSpeech K2
def transcribe_reazon_k2(model, audio_path):
    audio = k2_audio_from_path(audio_path)
    result = k2_transcribe(model, audio)
    return result.text

# Transcribe audio using ReazonSpeech ESPnet
def transcribe_reazon_espnet(model, audio_path):
    audio = espnet_audio_from_path(audio_path)
    result = espnet_transcribe(model, audio)
    return result.text

# Function to load models based on type
def load_model(model_name):
    if model_name.startswith("whisper_"):
        return whisper.load_model(model_name.split("_")[1])
    elif model_name == "reazonspeech_nemo":
        return load_reazon_nemo()
    elif model_name == "reazonspeech_k2":
        return load_reazon_k2()
    elif model_name == "reazonspeech_espnet":
        return load_reazon_espnet()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Evaluate CER and WER for a single model
def evaluate_model(model_name, audio_folder_path, transcript_file_path, max_files_to_process=10, do_normalize=True):
    """
    Evaluate CER and WER for a given model.

    Args:
        model_name (str): The name of the ASR model to evaluate.
        audio_folder_path (str): Path to the folder containing audio files.
        transcript_file_path (str): Path to the transcript file.
        max_files_to_process (int): Maximum number of files to process, allowing evaluation on a limited number of files instead of all available files.
        normalize (boolean): If True, normalize text by converting full-width to half-width and removing punctuation.
    """
    transcripts = load_transcripts(transcript_file_path)
    results = []

    print(f"Evaluating model: {model_name}")
    model = load_model(model_name)
    total_cer = 0.0
    total_wer = 0.0
    count = 0

    for idx, filename in enumerate(os.listdir(audio_folder_path)):
        if filename.endswith(".wav") and idx < max_files_to_process:
            audio_id = os.path.splitext(filename)[0]
            audio_path = os.path.join(audio_folder_path, filename)

            if audio_id in transcripts:
                ground_truth = transcripts[audio_id]
                try:
                    if model_name.startswith("whisper_"):
                        recognized_text = transcribe_whisper(model, audio_path)
                    elif model_name == "reazonspeech_nemo":
                        recognized_text = transcribe_reazon_nemo(model, audio_path)
                    elif model_name == "reazonspeech_k2":
                        recognized_text = transcribe_reazon_k2(model, audio_path)
                    elif model_name == "reazonspeech_espnet":
                        recognized_text = transcribe_reazon_espnet(model, audio_path)
                    else:
                        continue

                    if do_normalize:
                        ground_truth = normalize(ground_truth)
                        recognized_text = normalize(recognized_text)

                    ground_truth_tokenized = tokenize_text(ground_truth)
                    recognized_text_tokenized = tokenize_text(recognized_text)

                    cer_score = cer(ground_truth, recognized_text)
                    wer_score = wer(ground_truth_tokenized, recognized_text_tokenized)

                    total_cer += cer_score
                    total_wer += wer_score
                    count += 1

                    results.append({
                        "Model": model_name,
                        "Audio ID": audio_id,
                        "CER": cer_score,
                        "WER": wer_score,
                        "Ground Truth": ground_truth,
                        "Recognized Text": recognized_text
                    })
                except Exception as e:
                    print(f"Error processing {audio_id} with {model_name}: {e}")

    average_cer = total_cer / count if count > 0 else 0
    average_wer = total_wer / count if count > 0 else 0
    print("\n", end="")
    print(f"Average CER for {model_name}: {average_cer:.2%}")
    print(f"Average WER for {model_name}: {average_wer:.2%}")
    print("\n", end="")

    # Save results to CSV and download immediately
    df = pd.DataFrame(results)
    csv_filename = f"{model_name}_evaluation_results.csv"
    df.to_csv(csv_filename, index=False)
    # files.download(csv_filename)

# Define models
whisper_models = ["whisper_tiny", "whisper_base", "whisper_small", "whisper_medium", "whisper_large"]
reazonspeech_models = ["reazonspeech_nemo", "reazonspeech_k2", "reazonspeech_espnet"]

# %%
# Evaluate Whisper models
print("Evaluating Whisper models...")
for model in whisper_models:
    evaluate_model(model, audio_folder_path, transcript_file_path, max_files_to_process=10, do_normalize=True)

# %%
# Evaluate ReazonSpeech models
print("Evaluating ReazonSpeech models...")
for model in reazonspeech_models:
    evaluate_model(model, audio_folder_path, transcript_file_path, max_files_to_process=10, do_normalize=True)


