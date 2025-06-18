import moviepy as mv
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
import nltk.tokenize
import semantic_text_splitter
from tokenizers import Tokenizer
import pipeline
audioready = False
textready = False
ffmpeg_path = get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_path)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

path = ""
Vdir = ""
Vname = ""
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
    Vname = os.path.basename(path)
    return Vdir, Vname , path


def video2audio(Vpath , Apath):
    Apath = os.path.splitext(Vpath)[0] + ".wav"
    ffmpeg.input(Vpath).output(
        str(Apath),
        format = 'wav',
        acodec = 'pcm_s16le',
        ac = 1,
        ar = '16000'
    ).overwrite_output().run(quit=True)
    audioready = True
    return {Apath : str}


def audio2text(audioPath, language ="auto"):
    model = whisper.load_model('base')
    if language == "auto":
        transcribtion = model.transcribe(audioPath)
    else:
        transcribtion = model.transcribe(audioPath, language = language, task= "translate")
    textready = True
    return transcribtion


def summary(transcribtion: str , ratio : int):
    cleanText = transcribtion
    maxTokens = 1024
    summaries = []
    #modelName = "philschmid/bart-large-cnn-samsum"
    modelName = "sshleifer/distilbart-cnn-12-6"
    sumTokenizer = transformers.BartTokenizer.from_pretrained(modelName)
    model = transformers.BartForConditionalGeneration.from_pretrained(modelName)
    splitTokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = semantic_text_splitter.TextSplitter.from_huggingface_tokenizer(splitTokenizer, maxTokens)
    chunks = splitter.chunks(cleanText)
    cleaner = pipeline("text2text-generation" , model = "flan-t5-base")

    for chunk in chunks:
        cleanPrompt = f"Clean this transcript to be good for reading not dialoge or script {chunk}"
        cleaned = cleaner(cleanPrompt , max_length = 512 , do_sample = False)[0]['generated text']
        cleanText = []
        cleanText.append(cleaned)

    for paragraph in cleanText:
        wordsNumber = len(paragraph.split())
        maximumOut = int(wordsNumber/ratio + (((1/ratio)*100)/15))
        minimumOut = int(wordsNumber/ratio)
        inputs = sumTokenizer.encode("summarize :" + paragraph, return_tensors = "pt", max_length =1024 , trancution = True)
        summaryIdes = model.generate(inputs , max_length = maximumOut , min_length = minimumOut, length_penality = 1 , num_beams = 4 , early_stopping = True)
        Fsummary = sumTokenizer.decode(summaryIdes[0] , skip_special_tokens = True)
        summaries.append(Fsummary)
    FFsummary = " ".join(summaries)
    typer.echo("your summary:\n" + FFsummary)
    return FFsummary

         

def paraphrase(transcribtion:str):
    maxTokens = 1024
    tokenizer = transformers.AutoTokenizer.from_pretrained("flan-t5-base")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("flan-t5-base")
    splitTokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = semantic_text_splitter.TextSplitter.from_huggingface_tokenizer(splitTokenizer, maxTokens)
    chunks = splitter.chunks(transcribtion)
    Fchunks = []

    for chunk in chunks:
        prompt = f"""
You are an educational content assistant.

Rewrite the following video transcript chunk to create an engaging and stuctured explaination file.

Adapt your writing style based on the type of video content
- For Science or technical topics: use clear headings, bullet points, and explainations.
- For grammer or language: explain rules, give examples, and highlight structure.
- For math or physics problem solving: present the explaination step by step with labled equations and reasoning.
- For computer Science: explain the base and topics, with code blocks

Format the result like educational notes or articles with a clear titles,structured sections, and coherent flow.

transcribt:
\"\"\"
{chunk}
\"\"\"
"""
        inputs = tokenizer(prompt, return_tensors = "pt" , truncation = True)
        output = model.generate(**inputs,max_new_tokens = 1024)
        decoded = tokenizer.decode(output[0],skip_special_tokens = True)
        Fchunks.append(decoded.strip())
    Fparaphresed = "\n\n".join(Fchunks)
    return Fparaphresed



def transcription(audio, language:str = "auto"):
    model = whisper.load_model('base')
    if language == "auto":
        transcribtion = model.transcribe(audio)
    else:
        transcribtion = model.transcribe(audio, language = language, task = "translate")
    filePath = os.path.join(Vdir, "subtitle.srt")
    segments = transcribtion["segments"]
    for segment in segments:
        sTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        fTime = str(0)+str(timedelta(seconds=int(segment['end']))) +',000'
        text = segment['text']
        subtitleSeg = f"{segment['id']+1}\n{sTime} --> {fTime}\n{text}\n\n"
        with open(filePath , 'a' , encoding='utf-8') as srtFile:
            srtFile.write(subtitleSeg)
        baseName , extention = os.path.splitext(Vname)
        output = baseName + "_with_subtitles" + extention
        ffmpeg.input(path).filter('subtitles', srtFile).output(output).run(overwrite_output=True)
        

app = typer.Typer()
@app.command()
def video(Vpath: str, url : bool = typer.Option(False, help="Is this a URL?")):
    if url :
        link = Vpath
        downloadVideo(link)
    else :
        path = Vpath
        Vdir = os.path.dirname(path)
        Vname = os.path.basename(path)
        return path, Vdir, Vname
    

@app.command()
def addSub(lang : str = typer.Option("auto" , "--lang" , "-l"), help="""language is auto the orginal for translate use ISO 639-1 language code
Language	Code
Abkhazian	ab
Afar	aa
Afrikaans	af
Albanian	sq
Amharic	am
Arabic	ar
Armenian	hy
Assamese	as
Aymara	ay
Azerbaijani	az
Bashkir	ba
Basque	eu
Bengali, Bangla	bn
Bhutani	dz
Bihari	bh
Bislama	bi
Breton	br
Bulgarian	bg
Burmese	my
Byelorussian	be
Cambodian	km
Catalan	ca
Chinese	zh
Corsican	co
Croatian	hr
Czech	cs
Danish	da
Dutch	nl
English, American	en
Esperanto	eo
Estonian	et
Faeroese	fo
Fiji	fj
Finnish	fi
French	fr
Frisian	fy
Gaelic (Scots Gaelic)	gd
Galician	gl
Georgian	ka
German	de
Greek	el
Greenlandic	kl
Guarani	gn
Gujarati	gu
Hausa	ha
Hebrew	iw
Hindi	hi
Hungarian	hu
Icelandic	is
Indonesian	in
Interlingua	ia
Interlingue	ie
Inupiak	ik
Irish	ga
Italian	it
Japanese	ja
Javanese	jw
Kannada	kn
Kashmiri	ks
Kazakh	kk
Kinyarwanda	rw
Kirghiz	ky
Kirundi	rn
Korean	ko
Kurdish	ku
Laothian	lo
Latin	la
Latvian, Lettish	lv
Lingala	ln
Lithuanian	lt
Macedonian	mk
Malagasy	mg
Malay	ms
Malayalam	ml
Maltese	mt
Maori	mi
Marathi	mr
Moldavian	mo
Mongolian	mn
Nauru	na
Nepali	ne
Norwegian	no
Occitan	oc
Oriya	or
Oromo, Afan	om
Pashto, Pushto	ps
Persian	fa
Polish	pl
Portuguese	pt
Punjabi	pa
Quechua	qu
Rhaeto-Romance	rm
Romanian	ro
Russian	ru
Samoan	sm
Sangro	sg
Sanskrit	sa
Serbian	sr
Serbo-Croatian	sh
Sesotho	st
Setswana	tn
Shona	sn
Sindhi	sd
Singhalese	si
Siswati	ss
Slovak	sk
Slovenian	sl
Somali	so
Spanish	es
Sudanese	su
Swahili	sw
Swedish	sv
Tagalog	tl
Tajik	tg
Tamil	ta
Tatar	tt
Tegulu	te
Thai	th
Tibetan	bo
Tigrinya	ti
Tonga	to
Tsonga	ts
Turkish	tr
Turkmen	tk
Twi	tw
Ukrainian	uk
Urdu	ur
Uzbek	uz
Vietnamese	vi
Volapuk	vo
Welsh	cy
Wolof	who
Xhosa	xh
Yiddish	ji
Yoruba	yo
Zulu	zu
"""):
    if audioready:
        pass
    else:
        typer.echo("start converting to audio......")
        video2audio(path , Vdir)
    typer.echo("start creating subtitles file......")
    transcription(Apath, lang)
    typer.echo("video ready with subtitles......")

@app.command()
def summarize(ratio : int = 7,lang : str =typer.Option("auto" , "--lang","-l",help ="""language code and the default is video Orginal for translate use ISO 639-1 language code
Language	Code
Abkhazian	ab
Afar	aa
Afrikaans	af
Albanian	sq
Amharic	am
Arabic	ar
Armenian	hy
Assamese	as
Aymara	ay
Azerbaijani	az
Bashkir	ba
Basque	eu
Bengali, Bangla	bn
Bhutani	dz
Bihari	bh
Bislama	bi
Breton	br
Bulgarian	bg
Burmese	my
Byelorussian	be
Cambodian	km
Catalan	ca
Chinese	zh
Corsican	co
Croatian	hr
Czech	cs
Danish	da
Dutch	nl
English, American	en
Esperanto	eo
Estonian	et
Faeroese	fo
Fiji	fj
Finnish	fi
French	fr
Frisian	fy
Gaelic (Scots Gaelic)	gd
Galician	gl
Georgian	ka
German	de
Greek	el
Greenlandic	kl
Guarani	gn
Gujarati	gu
Hausa	ha
Hebrew	iw
Hindi	hi
Hungarian	hu
Icelandic	is
Indonesian	in
Interlingua	ia
Interlingue	ie
Inupiak	ik
Irish	ga
Italian	it
Japanese	ja
Javanese	jw
Kannada	kn
Kashmiri	ks
Kazakh	kk
Kinyarwanda	rw
Kirghiz	ky
Kirundi	rn
Korean	ko
Kurdish	ku
Laothian	lo
Latin	la
Latvian, Lettish	lv
Lingala	ln
Lithuanian	lt
Macedonian	mk
Malagasy	mg
Malay	ms
Malayalam	ml
Maltese	mt
Maori	mi
Marathi	mr
Moldavian	mo
Mongolian	mn
Nauru	na
Nepali	ne
Norwegian	no
Occitan	oc
Oriya	or
Oromo, Afan	om
Pashto, Pushto	ps
Persian	fa
Polish	pl
Portuguese	pt
Punjabi	pa
Quechua	qu
Rhaeto-Romance	rm
Romanian	ro
Russian	ru
Samoan	sm
Sangro	sg
Sanskrit	sa
Serbian	sr
Serbo-Croatian	sh
Sesotho	st
Setswana	tn
Shona	sn
Sindhi	sd
Singhalese	si
Siswati	ss
Slovak	sk
Slovenian	sl
Somali	so
Spanish	es
Sudanese	su
Swahili	sw
Swedish	sv
Tagalog	tl
Tajik	tg
Tamil	ta
Tatar	tt
Tegulu	te
Thai	th
Tibetan	bo
Tigrinya	ti
Tonga	to
Tsonga	ts
Turkish	tr
Turkmen	tk
Twi	tw
Ukrainian	uk
Urdu	ur
Uzbek	uz
Vietnamese	vi
Volapuk	vo
Welsh	cy
Wolof	who
Xhosa	xh
Yiddish	ji
Yoruba	yo
Zulu	zu""") , help="How much the summary smaller than orginal text"):
    if audioready:
        pass
    else:
        typer.echo("start converting to audio......")
        video2audio(path , Vdir)
    if textready:
        pass
    else:
        typer.echo("start transcribing......")
        audio2text(Apath)
    typer.echo("start summarizing......")
    summarize(transcribtion , ratio)

@app.command()
def explain():
    if audioready:
        pass
    else:
        typer.echo("start converting to audio......")
        video2audio(path , Vdir)
    if textready:
        pass
    else:
        typer.echo("start transcribing......")
        audio2text(Apath)
    typer.echo("start paraphrasing......")
    paraphrase(transcribtion)
    

def main():
    app
if __name__ == '__main__' :
    app
