# tokyo_whisperers
 This is the final project for Georgia Tech's Deep Learning course (CS 7643). We aim to enhance Japanese Automatic Speech Recognition (ASR) by fine-tuning OpenAI’s Whisper model on Japanese-specific data, assessing whether it can outperform the monolingual ReazonSpeech models in accuracy.


# Getting Started

## Local Setup (Linux)
1. Install conda and poetry
2. Create env
```
conda create -n "tokyo_whisperers" python=3.10
conda activate tokyo_whisperers
poetry run pip install cython ipykernel
poetry run pip install --no-use-pep517 youtokentome
poetry install
poetry run pip install sherpa-onnx==1.10.16+cuda -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
cp .env.example .env
# fill in the values in .env
```
3. Run the script
```
sh run.sh
```

## Google Colab Setup
Follow these steps to set up the environment to run the code in google colab:

### 1. Clone the repository and navigate to the directory
- Clone the repository's main branch
- Move to the directory:

```
!git clone https://github.com/ryujimorita/tokyo_whisperers.git
%cd tokyo_whisperers
```

### 2. (Optional) Create a new branch if you intend to make changes
```
!git checkout -b <new-branch-name>
```

### 3. Install Poetry (with the --pre argument) & install dependencies
- Install Poetry (with --pre command)
- Set the PATH environment variable for Poetry
- Install required dependencies:
```
!pip install -q --pre poetry
!export PATH="/root/.local/bin:$PATH"
!poetry run pip install cython setuptools wheel ipykernel
!poetry run pip install --no-use-pep517 youtokentome
!poetry install --no-root
```

### 4. Add Wandb credentials
```
!touch .env
!echo $'WANDB_API_KEY="xxx"\nWANDB_PROJECT="tokyo_whisperers"' >.env
!cat .env
```
Make sure to change `--wandb_run_name` argument in `run.sh` as well.

### 5. Execute the shell script
Run the provided shell script:
```
!sh run.sh
```

