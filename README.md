# **textStudy**
A study tool for who don't love videos, by converting into text files (markdown, LaTex, or pdf), summarizing, or preparing questions. works on local videos/ videos url
## Table of Content
- ### **[Installation](#installation)**
- ### **[Commands](#commands-1)**

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
pip install -e --break-system-packages
```
#### *that requires python, pip, and git*
#### if not avalible follow the next
----
#### for Windows

Go to  https://www.python.org/downloads/windows/ for python
    download the lastest version

and https://git-scm.com/download/win for git

#### for Mac

run in terminal
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
then
```bash
brew install python git
```

#### for linux

```bash
# Debian/Ubuntu
sudo apt install python3 python3-pip git -y

# Fedora/RHEL/CentOS/Rocky
sudo dnf install python3 python3-pip git -y

# Arch/Manjaro
sudo pacman -S python3 python3-pip git

# openSUSE
sudo zypper install python3 python3-pip git

# Solus
sudo eopkg install -y python3 python3-pip git
```

## Commands

### add the video
add the target video path or url to download

```bash
textStudy video path/to/video.mp4

textStudy video video-url  --url      
```

- To use online video instead of local video add `--url` or `-u`



## explain
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