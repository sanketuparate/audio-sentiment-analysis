import os
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import seaborn as sns
import librosa as lr
import librosa
import speech_recognition as sr
import IPython.display as ipd
import librosa.display
import warnings

warnings.filterwarnings("ignore")
import pickle

from flask import Flask, render_template, request, redirect
import speech_recognition as sr

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route("/recording", methods=["GET", "POST"])
def record():
    r = sr.Recognizer()
    transcript = ""

    with sr.Microphone() as source:
        print("Hiii SARK's Say something!")
        audio = r.record(source, duration=4)
    print("Recording Done")
    # write audio to a WAV file
    with open("static/output.wav", "wb") as f:
        f.write(audio.get_wav_data())
        f.close()
    print("Record saved!")

    return render_template('/index.html')


@app.route("/at", methods=["GET", "POST"])
def audiototext():
    r = sr.Recognizer()
    transcript = ""
    # Audio to text
    txt = sr.AudioFile("static/output.wav")

    with txt as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        print("You Said  : " + s)
    except Exception as e:
        s = "cannot recognize your voice"

    return render_template('/index.html', transcript=s)


x = []
y = []


def extract_feature(data, sampling_rate):
    result = np.array([])
	# Path=i
	# result=np.hstack((result, Path))

    stft = np.abs(librosa.stft(data))
    chromagram = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T, axis=0)
    result = np.hstack((result, chromagram))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfcc))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sampling_rate).T, axis=0)

    result = np.hstack((result, mel))

    return result


@app.route("/predict", methods=["GET", "POST"])
def predict():
    r = sr.Recognizer()
    label = ""
    Pkl_Filename = "new_mlp_74.pkl"
    # Loading the Model back from file
    with open(Pkl_Filename, 'rb') as file:
        model = pickle.load(file)

    x = []
    # for i in range (0,20,1):
    data, sampling_rate = librosa.load('static/output.wav')
    feature = extract_feature(data, sampling_rate)
    x.append(feature)

    l = model.predict(x)

    return render_template('/index.html', label=l)


def seq(y):
    from keras.models import model_from_json
    json_file = open('Final_Emotion_Detection_by_Sequential.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("saved_models/Final_Emotion_Detection_by_Sequential.h5")
    print("Loaded model from disk")
    return loaded_model.predict(y)


@app.route("/predictbyseq", methods=["GET", "POST"])
def predictbyseq():
 	r = sr.Recognizer()
 	label2 = ""
 	x=[]
 	#for i in range (0,20,1):
 	data, sampling_rate = librosa.load('static/output.wav')
 	feature=extract_feature(data,sampling_rate)
 	x.append(feature)

 	from sklearn.preprocessing import StandardScaler
 	scaler = StandardScaler()
 	y = scaler.fit_transform(x)
 	op=np.expand_dims(y,-1)
 	prediction = seq(op)
 	prediction=prediction.argmax(axis=1)


 	if prediction[0]==0:
 		label='Neutral'

 	elif prediction[0]==1:
 		label='Calmmmmmmm'

# 	elif prediction[0]==2:
# 		label='Happpuiuiui'

# 	elif prediction[0]==3:
# 		label='Sed.'

# 	elif prediction[0]==4:
# 		label='ANgrie hmph!'

# 	elif prediction[0]==5:
# 		label='f f f f fear'

# 	elif prediction[0]==6:
# 		label='Raaaahul? Correct!'

# 	elif prediction[0]==7:
# 		label='SURPRISSEEEE'


# 	return render_template('/index.html',label2=label)


@app.route("/request", methods=["GET", "POST"])
def index():
    transcript2 = ""
    print("I am innnn")
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        file.filename = "static/output.wav"  # some custom file name that you want
        file.save(file.filename)

        if file.filename == "":
            return redirect(request.url)
        if file:
            r = sr.Recognizer()
            txt = sr.AudioFile('static/output.wav')
            with txt as source:
                audio = r.record(source)
            try:
                transcript2 = r.recognize_google(audio)
                print("You Said  : " + transcript2)
            except Exception as e:
                transcript2 = "cannot recognize your voice"

    return render_template('index.html', transcript2=transcript2)


@app.route("/predictbyupload", methods=["GET", "POST"])
def predictbyupload():
    r = sr.Recognizer()
    l = ""
    Pkl_Filename = "new_mlp_74.pkl"
    # Loading the Model back from file
    with open(Pkl_Filename, 'rb') as file:
        model = pickle.load(file)

    x = []
    # res = requests.post(url, content)
    data, sampling_rate = librosa.load('static/output.wav')
    feature = extract_feature(data, sampling_rate)
    x.append(feature)

    l = model.predict(x)

    return render_template('/index.html', l=l)


@app.route("/playmusic", methods=["GET", "POST"])
def playwav():
    # from playsound import playsound
    # import gi
    # playsound('output.wav')
    # ipd.display(ipd.Audio('static/output.wav',autoplay=True))
    return render_template('/index.html')


# @app.route('/remove', methods=['POST'])
# def removefile():
# 	import os
# 	if os.path.exists("static/output.wav"):
# 		os.remove("static/output.wav")
# 		print("file remove successfully!!!")
# 	else:
# 		print("The file does not exist")
# 	return render_template('/index.html')


# @app.route("/wav")
# def streamwav():
#     def generate():
#         with open("/static/output.wav", "rb") as fwav:
#             data = fwav.read(1024)
#             while data:
#                 yield data
#                 data = fwav.read(1024)
#     return Response(generate(), mimetype="audio/x-wav")
# return render_template('/index.html')

if __name__ == '__main__':
    app.run(host="localhost", debug=True)
# app.run()
