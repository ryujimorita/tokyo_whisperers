# tokyo_whisperers
 This is the final project for Georgia Tech's Deep Learning course (CS 7643). We aim to enhance Japanese Automatic Speech Recognition (ASR) by fine-tuning OpenAI’s Whisper model on Japanese-specific data, assessing whether it can outperform the monolingual ReazonSpeech models in accuracy.


# Getting Started
1. Install conda and poetry
2. Create env
```
conda create -n "tokyo_whisperers" python=3.10
conda activate tokyo_whisperers
poetry run pip install cython
poetry run pip install --no-use-pep517 youtokentome
poetry install
```
3. Run the script
```
sh run.sh
```
