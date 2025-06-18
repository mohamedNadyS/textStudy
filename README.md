# **textStudy**
A study tool for who don't love videos, by converting into text, summarizing, or preparing tests and flash cards. works on youtube videos, local videos/ audios
## Table of Content
- ### **[Installation](#installation)**
- ### **[Commands](#commands-1)**
- ###

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

#### add subtitles
```bash
textStudy -addSub --lang (optional in iso 639-1 language code, default video language)
```

#### summarization
```bash
textStudy -summary --ratio     #How much smaller than orginal 
```

#### re-explain
```bash
textStudy
```