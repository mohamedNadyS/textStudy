# **textStudy** - Video Content Analysis Tool
A study tool for who don't love videos, by converting into educational materials (markdown, LaTex, or pdf), summarizies, or preparing questions. works on local videos/ videos url by Advanced AI models, and APIs

## Table of Content
- ### **[Features](#features)**
- ### **[Installation](#installation)**
- ### **[Commands](#commands-1)**


## 📋 Features

### Core Functions
1. **Video Processing**: Download and process videos from URLs or local files
2. **Audio Extraction**: Convert video to audio for transcription
3. **Transcription**: Generate accurate text transcripts using Whisper
4. **Subtitles**: Add embedded subtitles to videos
5. **Summarization**: Create intelligent summaries with configurable ratios
6. **Educational Content**: Generate comprehensive educational materials
7. **Assessment Questions**: Create diverse question types for learning

### Output Formats
- **Markdown**: Well-structured educational content
- **PDF**: Professional printable documents
- **LaTeX**: Academic document formatting
- **Text**: Plain text summaries and transcripts

### Question Types
- Multiple Choice Questions (MCQ)
- True/False Questions
- Short Answer Questions
- Essay Questions
- Fill-in-the-Blank Questions


## Installation

#### Clone the repository
```bash
git clone https://github.com/mohamedNadyS/textStudy.git
```
#### Open the Directory
```bash
cd textStudy
```
#### install the requirments
```bash
pip install sentencepiece --prefer-binary
pip install . --break-system-packages
```
#### *that requires >=3.8,<3.12 python , pip, pyenv, ffmpeg, and git*
#### if not avalible follow the next
----
#### for Windows
python if want use a virual enviroment:
    Go to  https://github.com/pyenv-win/pyenv-win#installation for python, pip, and pyenv
        download between 3.8 and 3.11 version
    then in powershell run
```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -e .
```
or
Go to https://www.python.org/downloads/windows/ and download 3.11.9 version
, https://ffmpeg.org/download.html for ffmpeg
, and https://git-scm.com/download/win for git



#### for Mac

run in terminal
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

```
then
```bash
brew install pyenv git ffmpeg
pyenv install 3.11.9
pyenv local 3.11.9

python -m venv venv
source venv/bin/activate

pip install -U pip
pip install -e .
```

#### for linux

```bash
# Debian/Ubuntu
sudo apt install python3.11 python3.11-venv python3.11-distutils git ffmpeg -y
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Fedora/RHEL/CentOS/Rocky
sudo dnf install python3.11 python3.11-pip git ffmpeg -y

# Arch/Manjaro Must use pyenv
sudo pacman -S --needed base-devel git openssl zlib xz tk
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# openSUSE
sudo zypper install python3.11 python3.11-pip git ffmpeg


# to creat enviroment if python 3.11 if this didn't succes in all this systems run the next

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install . --break-system-packages  # or just `pip install .` if not root

```

## Commands

### add the video
---
add the target video path or url to download

```bash
textStudy video path/to/video.mp4

textStudy video video-url  --url      
```

- To use online video instead of local video add `--url` or `-u`



## explain
---
creates a file (markdown, pdf, or LaTeX) that paraphrase for video content

```bash
textStudy explain --type pdf --onAPI --api huggingface --model flan-t5-base --key (API_key)
```
- To determine the output file type use `--type`/`-t` then type from "pdf, markdown, and LaTeX" (required Argument)
- To use API (optional) instead of download model locally (automatically in the program) for weak devices or who want to use a specific models or more powerful model use the following argumnet
    - `--onAPI` or `-o` to enable API mode
    -  `--api` or `-a` to enter the API name options are: `huggingface, cohere, replicate, groq, ollama, or gemini`
    - `--model` or `-m` to enter the model name like `flan-t5-base` the model of offline mode
    - `--key` or `-k` to enter your API key

### preparing questions
---
creats a question file (markdown, odf, or LaTeX) on the video content by any question style jsut as mcq, short anwer, essay question, etc

```bash
textStudy questions --number 20 --style mcq --type LaTeX --tAPI --api huggingface --model flan-t5-base --key (API_key)
```

- To determine the number of generated questions use `--number` or `-n` (required Argument)
- To determine the style of question use `--style` or `-s` (required Argument)
- To determine the output file type use `--type`/`-t` then type from "pdf, markdown, and LaTeX" (required Argument)
- To use API (optional) instead of download model locally (automatically in the program) for weak devices or who want to use a specific models or more powerful model use the following argumnet
    - `--onAPI` or `-o` to enable API mode
    -  `--api` or `-a` to enter the API name options are: `huggingface, cohere, replicate, groq, ollama, or gemini`
    - `--model` or `-m` to enter the model name like `flan-t5-base` the model of offline mode
    - `--key` or `-k` to enter your API key

### add subtitles
---
create subtitles file and combine with the video
```bash
textStudy addSub --lang en
```

- To translate (Optional) the text into any language (the default is video language) use  `--lang` or `-l` then the iso 639-1 language code like `en` for english

### summarization
---
creates a readable, comprehensive summary of the video
```bash
textStudy summarize --ratio 8 --lang en         

```

- To determine how many times the summary is smaller than the original use `--ratio` or `r` then integer `8` for example (required Argument)
- To translate (Optional) `--lang` or `-l` then iso 639-1 language code like `en`
