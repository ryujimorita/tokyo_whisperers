# tokyo_whisperers
 This is the final project for Georgia Tech's Deep Learning course (CS 7643). We aim to enhance Japanese Automatic Speech Recognition (ASR) by fine-tuning OpenAI’s Whisper model on Japanese-specific data, assessing whether it can outperform the monolingual ReazonSpeech models in accuracy.


# Getting Started

## Local Setup (Linux)
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

## Google Colab Setup
Follow these steps to set up the environment to run the code in google colab:

### 1. Clone the repository and navigate to the directory
Run the following commands to clone the repository and change to its directory:

```
!git clone https://github.com/ryujimorita/tokyo_whisperers.git
%cd tokyo_whisperers
```

### 2. (Optional) Create a new branch if you intend to make changes
```
!git checkout -b <new-branch-name>
```

### 3. Install Poetry (with the --pre argument)
Use the commands below to install Poetry and set the PATH environment variable:
```
!pip install -q --pre poetry
!export PATH="/root/.local/bin:$PATH"
```

### 4. Install dependencies
Run the following commands to install required dependencies:
```
!poetry run pip install cython setuptools wheel
!poetry run pip install --no-use-pep517 youtokentome
!poetry install --no-root
```

### 5. Execute the shell script
Finally, run the provided shell script:
```
!sh run.sh
```

