import moviepy as mv
import speech_recognition as sr
import typer
import whisper
import os
from imageio_ffmpeg import get_ffmpeg_exe
import ffmpeg
import transformers
from pathlib import Path
import yt_dlp
from datetime import datetime, timedelta
import uuid
from rich.progress import track
import nltk
ffmpeg_path = get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_path)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

path = ""
Vdir = ""

class progressHook:
    def __init__(self , progress , task_id):
        self.progress = progress
        self.task_id = task_id

    def __call__(self , d):
        if d['state'] == 'downloading':
            if 'total_bytes' in d:
                total = d['total_bytes']
                downloaded = d['downloaded_bytes']
                self.progress.update(self.task_id , total = total , completed = downloaded)
        elif 'total_bytes_estimate' in d:
            total = d['total_bytes_estimate']
            downloaded = d['downloaded_bytes']
            self.progress.update(self.task_id , total = total , completed = downloaded)
        elif d['state'] == 'finished':
            self.progress.update(self.task_id , completed = d.get('total_bytes' , 100) , total = d.get('total_bytes' , 100))



def downloadVideo(Vlink):
    dir = Path.cwd()
    Vdir = dir / "videos"
    Vdir.mkdir(exist_ok=True)
    nowTime = datetime.now().strftime("%Y%m%d_%H%M%S")
    uID = uuid.uuid4().hex[:6]
    Vname = f"%(title).50s_{nowTime}_{uID}.%(ext)s"

    path = Vdir / Vname
    ydl_opts = {
        'outtmpl': str(path),
        'format': 'bestvideo[height<=480]+bestaudio/best',
        'merge_output_format': 'mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([Vlink])
        except:
            typer.echo("unavaliable URL")
    
    path = None
    for file in Vdir.glob(f"*{nowTime}_{uID}*"):
        if file.is_file() and file.suffix.lower() in ('.mp4' ,'.mkv' , '.avi' ,'.mov'):
            path = file
            break
    return Vdir, path.name, path
'''    return {
        'Fdir' : str(Vdir),
        'Fname' : path.name,
        'Fpath' : str(path)
    }
'''


def video2audio(Vpath , Apath):
    Apath = os.path.splitext(Vpath)[0] + ".wav"
    ffmpeg.input(Vpath).output(
        str(Apath),
        format = 'wav',
        acodec = 'pcm_s16le',
        ac = 1,
        ar = '16000'
    ).overwrite_output().run(quit=True)
    return {Apath : str}

def audio2text(audioPath):
    model = whisper.load_model('base')
    transcribtion = model.transcribe(audioPath)
    return transcribtion


def summary(transcribtion: str , ratio : int):
    modelName = "fa"
    pass




def transcription(audio, language):
    model = whisper.load_model('base')
    transcribtion = model.transcribe(audio)
    segments = transcribtion["segments"]
    for segment in segments:
        sTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        fTime = str(0)+str(timedelta(seconds=int(segment['end']))) +',000'
        text = segment['text']
        subtitleSeg = f"{segment['id']+1}\n{sTime} --> {fTime}\n{text}\n\n"
        with open("subtitle.srt" , 'a' , encoding='utf-8') as srtFile:
            srtFile.write(subtitleSeg)
        

app = typer.Typer()
@app.command()
def video(Vpath: str, url : bool = typer.Option(False, help="Is this a URL?")):
    if url :
        link = Vpath
        downloadVideo(link)
    else :
        path = Vpath
        return path
    

@app.command()
def addSub(lang : str = "en", help="""use for ISO 639-1 language code for language 
Language	Code
Abkhazian	AB
Afar	AA
Afrikaans	AF
Albanian	SQ
Amharic	AM
Arabic	AR
Armenian	HY
Assamese	AS
Aymara	AY
Azerbaijani	AZ
Bashkir	BA
Basque	EU
Bengali, Bangla	BN
Bhutani	DZ
Bihari	BH
Bislama	BI
Breton	BR
Bulgarian	BG
Burmese	MY
Byelorussian	BE
Cambodian	KM
Catalan	CA
Chinese	ZH
Corsican	CO
Croatian	HR
Czech	CS
Danish	DA
Dutch	NL
English, American	EN
Esperanto	EO
Estonian	ET
Faeroese	FO
Fiji	FJ
Finnish	FI
French	FR
Frisian	FY
Gaelic (Scots Gaelic)	GD
Galician	GL
Georgian	KA
German	DE
Greek	EL
Greenlandic	KL
Guarani	GN
Gujarati	GU
Hausa	HA
Hebrew	IW
Hindi	HI
Hungarian	HU
Icelandic	IS
Indonesian	IN
Interlingua	IA
Interlingue	IE
Inupiak	IK
Irish	GA
Italian	IT
Japanese	JA
Javanese	JW
Kannada	KN
Kashmiri	KS
Kazakh	KK
Kinyarwanda	RW
Kirghiz	KY
Kirundi	RN
Korean	KO
Kurdish	KU
Laothian	LO
Latin	LA
Latvian, Lettish	LV
Lingala	LN
Lithuanian	LT
Macedonian	MK
Malagasy	MG
Malay	MS
Malayalam	ML
Maltese	MT
Maori	MI
Marathi	MR
Moldavian	MO
Mongolian	MN
Nauru	NA
Nepali	NE
Norwegian	NO
Occitan	OC
Oriya	OR
Oromo, Afan	OM
Pashto, Pushto	PS
Persian	FA
Polish	PL
Portuguese	PT
Punjabi	PA
Quechua	QU
Rhaeto-Romance	RM
Romanian	RO
Russian	RU
Samoan	SM
Sangro	SG
Sanskrit	SA
Serbian	SR
Serbo-Croatian	SH
Sesotho	ST
Setswana	TN
Shona	SN
Sindhi	SD
Singhalese	SI
Siswati	SS
Slovak	SK
Slovenian	SL
Somali	SO
Spanish	ES
Sudanese	SU
Swahili	SW
Swedish	SV
Tagalog	TL
Tajik	TG
Tamil	TA
Tatar	TT
Tegulu	TE
Thai	TH
Tibetan	BO
Tigrinya	TI
Tonga	TO
Tsonga	TS
Turkish	TR
Turkmen	TK
Twi	TW
Ukrainian	UK
Urdu	UR
Uzbek	UZ
Vietnamese	VI
Volapuk	VO
Welsh	CY
Wolof	WO
Xhosa	XH
Yiddish	JI
Yoruba	YO
Zulu	ZU
"""):
    typer.echo("start converting to audio......")

    video2audio(path , Vdir)
    typer.echo("start creating subtitles file......")
    transcription("output.wav", lang)
    typer.echo("video ready with subtitles......")

@app.command()
def summarize(ratio : int = 7, help="How much the summary smaller than orginal text"):
    typer.echo("start converting to audio......")
    video2audio(path , Vdir)
    typer.echo("start transcribing......")
    audio2text()
    pass
    

def main():
    app
if __name__ == '__main__' :
    app