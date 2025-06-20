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
import semantic_text_splitter
from tokenizers import Tokenizer
import pipeline
from transformers import pipeline
import weasyprint
import requests
import textwrap
import json

audioready = False
textready = False
ffmpeg_path = get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_path)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

path = ""
Vdir = ""
Vname = ""
Apath = ""
transcribtion = ""
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

def markDown(text, functionU):
    mdPath = os.path.splitext(path)[0] + functionU + ".md"
    with open(mdPath,"w", encoding="utf-8") as m:
        m.write("#generated content\n\n")
        for i, line in enumerate(text.split("\n"), 1):
            if line.strip():
                m.write(f"{i}. {line.strip()}\n\n")



def pdf(text, functionU):
    pdfPath = os.path.splitext(path)[0]+functionU+".pdf"
    headd = "generated content"
    html = f"""
<html>
<head>
    <meta charset="utf-8">
    <style>
    body {{font-family: 'Georgia', serif; padding: 1em; border-radius: 5px; overflow: auto; }}
    h1 {{ text-align: center; color: #003366;}}
    h2 {{ color: #005588; margin-top: 1.5em; }}
    pre {{ background: #f4f4f4; padding: 1em; border-radius: 5px; overflow-x: auto; }}
    code {{ background: #f0f0f0; padding: 1em; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<h1>{headd}</h1>
<pre>{text.strip()}</pre>
</html>
"""
    weasyprint.HTML(string=html).write_pdf(pdfPath)
    pass


def LaTeX(text, functionU):
    latexPath = os.path.splitext(path)[0]+ functionU +".tex"
    with open(latexPath,"w", encoding="utf-8") as l:
        l.write(r"""\documentclass{article}
\usepackage[utf-8]{inputenc}
\usepackage{enumitem}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{geometry}
\geometry{margin=1in}
\title{Generated Content}
\date{\today}
\begin{document}
\maketitle
""")
        safeText = (
            text.replace("&",r"\&")
                .replace("%",r"\%")
                .replace("#",r"\#")
                .replace("_",r"\_")
                .replace("~",r"\textasciitilde{}")
                .replace("^",r"\textasciicircum{}")
                .replace("\\",r"\textbackslash{}")
        )
        l.write(safeText.strip())
        l.write(r"\end{document}")

def downloadVideo(Vlink):
    global path, Vdir, Vname
    dir = Path.cwd()
    Vdir = dir / "videos"
    Vdir.mkdir(exist_ok=True)
    nowTime = datetime.now().strftime("%Y%m%d_%H%M%S")
    uID = uuid.uuid4().hex[:6]
    Vname = f"%(title).50s_{nowTime}_{uID}.%(ext)s"

    temp_path = Vdir / Vname
    ydl_opts = {
        'outtmpl': str(temp_path),
        'format': 'bestvideo[height<=480]+bestaudio/best',
        'merge_output_format': 'mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([Vlink])
        except Exception as e:
            typer.echo(f"unavaliable URL: {e}")
            return None,None,None

    
    path = None
    for file in Vdir.glob(f"*{nowTime}_{uID}*"):
        if file.is_file() and file.suffix.lower() in ('.mp4' ,'.mkv' , '.avi' ,'.mov'):
            path = str(file)
            break
    if path:
        Vname = os.path.basename(path)
        return str(Vdir), Vname , path
    else:
        typer.echo("Downloaded file not found")
        return None,None,None


def video2audio(Vpath , Apath):
    global audioready
    Apath = os.path.splitext(Vpath)[0] + ".wav"
    ffmpeg.input(Vpath).output(
        str(Apath),
        format = 'wav',
        acodec = 'pcm_s16le',
        ac = 1,
        ar = '16000'
    ).overwrite_output().run(quit=True)
    audioready = True
    return Apath


def audio2text(audioPath, language ="auto"):
    global textready, transcribtion
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
        inputs = sumTokenizer.encode("summarize :" + paragraph, return_tensors = "pt", max_length =1024 , truncation = True)
        summaryIdes = model.generate(inputs , max_length = maximumOut , min_length = minimumOut, length_penalty = 1 , num_beams = 4 , early_stopping = True)
        Fsummary = sumTokenizer.decode(summaryIdes[0] , skip_special_tokens = True)
        summaries.append(Fsummary)
    FFsummary = " ".join(summaries)
    typer.echo("your summary:\n" + FFsummary)
    return FFsummary

         

def paraphrase(transcribtion:str, fileType, tAPI: bool= False, API:str = None, apiModel:str = None, apiKey:str =None):
    if fileType == "pdf":
        fftype ="plain text formatted for printable PDF."
    elif fileType == "markdown":
        fftype = """
- Format output in **Markdown**.
- Start each chunk with: `## {Section name}{n}`"""
    elif fileType == "latex":
        fftype = """
- Format output in **LaTeX**.
- Start each section with: `\\section*{name}`
- Use `\begin{enumerate}` for paragraph."""

    maxTokens = 1024
    
    if tAPI:
        chunks = textwrap.wrap(transcribtion, width = 1500,break_long_words=False)
    else:
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
with {fileType} format that {fftype}
transcribt:
\"\"\"
{chunk}
\"\"\"
"""
        if tAPI:
            if API.lower == "huggingface":
                headers = {"Authorization":f"Bearer {apiKey}"}
                url = f"https://api-interface.huggingface.co/models/{apiModel}"
                response = requests.post(url, headers=headers, json={"inputs":prompt})
                text = response.json()
                result = text[0]['generated_text'] if isinstance(text,list) else text.get("generated_text","")

            elif API.lower =="cohere":
                headers = {"Authorization":f"Bearer {apiKey}"}
                url = "https://api.cohere.ai/generate"
                payload = {"model":apiModel, "prompt": prompt, "max_tokens":800, "temperature":0.7}
                response = requests.post(url , headers=headers, json=payload)
                result = response.json()['generations'][0]['text']

            elif API.lower =="replicate":
                headers = {"Authorization":f"Token {apiKey}", "Content-Type":"application/json"}
                url = "https://api.replicate.com/v1/predictions"
                payload = {"model":apiModel, "input":{"prompt": prompt}}
                response = requests.post(url, headers= headers, json= payload)
                prediction = response.json()
                result =prediction.get("output", "[replicate pending...]")
            
            elif API.lower =="groq":
                headers = {"Autorization":f"Bearer {apiKey}"}
                url = "https://api.groq.com/openai/v1/chat/completions"
                payload = {"model":apiModel, "messages":[{"role":"user", "content": prompt}]}
                response = requests.post(url ,  headers= headers, json=payload)
                result= response.json()['choices'][0]['message']['content']

            elif API.lower == "ollama":
                url = "https://localhost:11434/api/generate"
                payload = {"model": apiModel,"prompt": prompt}
                response = requests.post(url, json= payload, stream=True)
                chunks = []
                for i in response.iter_lines():
                    if i:
                        part = json.loads(i.decode("utf-8"))
                        chunks.append(part.get("response",""))
                result = "".join(chunks)

            elif API.lower == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{apiModel}:generateContent?key={apiKey}"
                headers = {"content-type": "application/json"}
                payload = {"contents":[{"parts":[{"text":prompt}]}]}
                response = requests.post(url, headers=headers, json=payload)
                result =response.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            else: 
                raise ValueError(f"API '{API}' is not supported, true inputs, or free, or reached free-limit")
            Fchunks.append(result.strip())
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained("flan-t5-base")
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained("flan-t5-base")
            inputs = tokenizer(prompt, return_tensors = "pt" , truncation = True)
            output = model.generate(**inputs,max_new_tokens = 1024)
            decoded = tokenizer.decode(output[0])
            Fchunks.append(decoded.strip())


    Fparaphresed = "\n\n".join(Fchunks)
    if fileType == "pdf":
        pdf(Fparaphresed, "_paraphrased")
    elif fileType == "markdown":
        markDown(Fparaphresed, "_paraphrased")
    elif fileType == "latex":
        LaTeX(Fparaphresed, "_paraphrased")
    return Fparaphresed



def transcription(audio, language:str = "auto"):
    model = whisper.load_model('base')
    if language == "auto":
        transcribtion = model.transcribe(audio)
    else:
        transcribtion = model.transcribe(audio, language = language, task = "translate")
    filePath = os.path.join(Vdir, "subtitle.srt")
    segments = transcribtion["segments"]

    with open(filePath , 'a' , encoding='utf-8') as srtFile:
        for segment in segments:
            sTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
            fTime = str(0)+str(timedelta(seconds=int(segment['end']))) +',000'
            text = segment['text']
            subtitleSeg = f"{segment['id']+1}\n{sTime} --> {fTime}\n{text}\n\n"
            srtFile.write(subtitleSeg)

    baseName , extention = os.path.splitext(Vname)
    output = os.path.join(Vdir, baseName + "_with_subtitles" + extention)
    ffmpeg.input(path).filter('subtitles', filePath).output(output).run(overwrite_output=True)
    typer.echo(f"video with subtitles saved as: {output}")
        

def questions(numberQuestions ,transcribtion ,fileType, style:str = "mcq", tAPI:bool = False, API:str =None,apiModel:str=None,apiKey:str=None):
    def questionsPrompt(chunk:str, style:str ,Number : int):
        return f""" You are an intelligent quesions generator.

Analys the following text and generate {Number} {style} questions that cover the key ideas, facts, and reasoning presented.

Be sure the quesions cover the full content of the passage and assess undersainding fairly.

Text
\"\"\"
{chunk}
\"\"\"

Now generate {Number} {style} questions based on the above text.

{explainfileType}
"""
    if tAPI:
        chunks = textwrap.wrap(transcribtion, width = 1500, break_long_words=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        splitter = semantic_text_splitter.TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens = 1024)
        chunks = splitter.chunks(transcribtion)
    Ftokenizer = transformers.AutoTokenizer.from_pretrained("flan-t5-base")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("flan-t5-base")
    chunkNtokens = [len(tokenizer.encode(chunk))for chunk in chunks]
    TotalNtokens = sum(chunkNtokens) 
    quesionsForChunk = []
    connections = []
    if fileType == "markdown":
        explainfileType = """
- Format question in **markdwon**.
- start each chunk with: ##section
- Use bold for question numbers: `**1.**`
- For MCQ if question are MCQ list options using `- A)`,`- B)` ,etc.
"""
    elif fileType == "pdf":
        explainfileType = """
- Write in plain text formatted for printable PDF.
- Start each section with: `Section: Questions`
- Use numbered questions and A), B), C)... for MCQs.
"""
    elif fileType == "latex":
        explainfileType =r"""
- Format output in **LaTeX**.
- Start each section with: `\section*{Questions}`
- Use `\begin{enumerate}` for questions.
- Use `\item` for each question, and if MCQ, embed options using `\begin{itemize} \item A... \end{itemize}`
"""
    else:
        typer.echo("unavailable file type. choose from: markdown, pdf, latex", fg = "darkred")
        return None

    for tokens in chunkNtokens:
        quesions = max(1, round((tokens / TotalNtokens)* numberQuestions))
        quesions.append(quesionsForChunk)

    while sum(quesionsForChunk) > numberQuestions:
        maxChunk = quesionsForChunk.index.max(quesionsForChunk)
        maxChunk -=1
    while sum(quesionsForChunk) < numberQuestions:
        minChunk = quesionsForChunk.index.min(quesionsForChunk)
        minChunk += 1
    
    for chunkI, questionsN in zip(chunks, quesionsForChunk):
        prompt = questionsPrompt(chunkI , style , questionsN)
        if tAPI:
            API = API.lower()

            if API == "huggingface":
                headers = {"Authorization": f"Bearer {apiKey}"}
                url = f"https://api-inference.huggingface.co/models/{apiModel}"
                response = requests.post(url, headers=headers, json={"inputs": prompt})
                result = response.json()
                decoded = result[0]['generated_text'] if isinstance(result, list) else result.get("generated_text", "")

            elif API == "cohere":
                headers = {"Authorization": f"Bearer {apiKey}", "Content-Type": "application/json"}
                url = "https://api.cohere.ai/generate"
                payload = {
                    "model": apiModel,
                    "prompt": prompt,
                    "max_tokens": 800,
                    "temperature": 0.7
                }
                response = requests.post(url, headers=headers, json=payload)
                decoded = response.json()['generations'][0]['text']

            elif API == "replicate":
                headers = {"Authorization": f"Token {apiKey}", "Content-Type": "application/json"}
                url = "https://api.replicate.com/v1/predictions"
                payload = {
                    "version": apiModel,
                    "input": {"prompt": prompt}
                }
                response = requests.post(url, headers=headers, json=payload)
                result = response.json()
                decoded = result.get("output", "[replicate pending...]")

            elif API == "groq":
                headers = {"Authorization": f"Bearer {apiKey}"}
                url = "https://api.groq.com/openai/v1/chat/completions"
                payload = {
                    "model": apiModel,
                    "messages": [{"role": "user", "content": prompt}]
                }
                response = requests.post(url, headers=headers, json=payload)
                decoded = response.json()['choices'][0]['message']['content']

            elif API == "ollama":
                url = "http://localhost:11434/api/generate"
                payload = {"model": apiModel, "prompt": prompt}
                response = requests.post(url, json=payload, stream=True)
                parts = [json.loads(line.decode())['response'] for line in response.iter_lines() if line]
                decoded = "".join(parts)

            elif API == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{apiModel}:generateContent?key={apiKey}"
                headers = {"Content-Type": "application/json"}
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                response = requests.post(url, headers=headers, json=payload)
                decoded = response.json()["candidates"][0]["content"]["parts"][0]["text"]

            else:
                raise ValueError(f"API '{API}' is not supported or free.")
        else:
            inputs = Ftokenizer(prompt, return_tensors = "pt", trancation = True)
            outputs = model.generate(**inputs, max_new_tokens = 1024)
            decoded = Ftokenizer.decode(outputs[0])
        connections.append(decoded.strip())
        FfQuestions = "\n\n".join(connections)
        if fileType == "latex":
            LaTeX(FfQuestions, "_questions")
        elif fileType == "pdf":
            pdf(FfQuestions, "_questions")
        elif fileType == "md":
            markDown(FfQuestions, "_questions")
        return FfQuestions
    


app = typer.Typer()
@app.command()
def video(Vpath: str, url : bool = typer.Option(False, "--url", "-u",help="Is this a URL?")):
    global path, Vdir, Vname
    if url :
        link = Vpath
        result = downloadVideo(link)
        if result[0] is None:
            typer.echo("Failed to download video")
            return 
        Vdir, path, Vname = result

    else :
        path = Vpath
        Vdir = os.path.dirname(path)
        Vname = os.path.basename(path)
        typer.echo(f"video set: {Vname}")
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
    global audioready, Apath, path, Vdir
    if not path:
        typer.echo("Please set a video first using 'textStudy video <path>'")
        return
    if not audioready:
        typer.echo("start converting to audio......")
        Apath = video2audio(path , Vdir)
        if not Apath:
            return

    typer.echo("start creating subtitles file......")
    transcription(Apath, lang)
    typer.echo("video ready with subtitles......")

@app.command()
def summarize(ratio : int = typer.Option(6, "--ratio", "-r", help="How much the summary smaller than original text"),lang : str =typer.Option("auto" , "--lang","-l",help ="""language code and the default is video Orginal for translate use ISO 639-1 language code
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
Zulu	zu""")):
    global audioready, textready, Apath, path, Vdir, transcribtion
    if not path:
        typer.echo("Please set a video first using 'textStudy video <path>'")
        return

    if not audioready:
        typer.echo("start converting to audio......")
        Apath = video2audio(path , Vdir)
        if not Apath:
            return

    if not textready:
        typer.echo("start transcribing......")
        result = audio2text(Apath)
        if not result:
            return

    typer.echo("start summarizing......")
    summarize(transcribtion , ratio)

@app.command()
def explain(fileType: str = typer.Option(...,"-t","--type",help ="Output file markdown, LaTex, or pdf as you want, and will be saved in same your video directory"),onAPI:bool = typer.Option(False,"-o","--onAPI" ,help="Are you want work with API instead of locally?"),API:str=typer.Option(None,"-a","--api"),model:str=typer.Option(None,"-m","--model"), key:str=typer.Option(None,"-k","--key")):
    global audioready, textready, Apath, path, Vdir, transcribtion
    if not path:
        typer.echo("Please set a video first using 'textStudy video <path>'")
        return
    usedType = fileType.lower()
    if not audioready:
        typer.echo("start converting to audio......")
        Apath = video2audio(path , Vdir)
        if not Apath:
            return

    if not textready:
        typer.echo("start transcribing......")
        result = audio2text(Apath)
        if not result:
            return

    typer.echo("start paraphrasing......")
    paraphrase(transcribtion, usedType,onAPI,API,model,key)
    
@app.command()
def questions(number :int= typer.Option(...,"-n","--number", help="number generated questions on the video"), style:str = typer.Option(...,"-s","--style",help = "style of questions from next: multiple choice, true/false, short answer, essay, fill in the blank"), file_type: str = typer.Option(...,"-t","--type",help ="Output file markdown, LaTex, or pdf as you want, and will be saved in same your video directory"),tAPI:bool = typer.Option(False,"-t","--tAPI" ,help="Are you want work with API instead of locally?"),API:str=typer.Option(None,"-a","--api"),model:str=typer.Option(None,"-m","--model"), key:str=typer.Option(None,"-k","--key")):
    global audioready, textready, Apath, path, Vdir, transcribtion
    usedType = file_type.lower()
    if not path:
        typer.echo("Please set a video first using 'textStudy video <path>'")
        return
    if not audioready:
        typer.echo("start converting to audio......")
        Apath = video2audio(path , Vdir)
        if not Apath:
            return

    if not textready:
        typer.echo("start transcribing......")
        result = audio2text(Apath)
        if not result:
            return
    

    typer.echo("start preparing questions......")
    questions(number,transcribtion, usedType,style,tAPI,API,model,key)

main = app
if __name__ == '__main__' :
    app()
