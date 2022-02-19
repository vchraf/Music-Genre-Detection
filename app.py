from flask import Flask, render_template, url_for, request, redirect,send_file

import librosa
import librosa.feature
from pydub import AudioSegment

from sklearn.ensemble import RandomForestClassifier
import catboost as cb

import joblib
from os import path

import pandas as pd
import numpy as np
import os

app = Flask(__name__)

def MP32WAV(path):
  sound = AudioSegment.from_mp3(path)
  n_path = path+".wav"
  sound.export(n_path, format="wav")
  return n_path


def audio_pipeline(audio,sr):
  features = []
  #calcul chroma_stft
  chroma_stft = librosa.feature.chroma_stft(audio, sr)
  features.append(np.mean(chroma_stft))
  features.append(np.var(chroma_stft))
  
  #calcul RMS
  rms = librosa.feature.rms(audio)
  features.append(np.mean(rms))
  features.append(np.var(rms))  
  
  #calcul spectral_centroid
  spectral_centroid = librosa.feature.spectral_centroid(audio, sr)
  features.append(np.mean(spectral_centroid))
  features.append(np.var(spectral_centroid))
  
  #calcul spectral_bandwidth
  spectral_bandwidth = librosa.feature.spectral_bandwidth(audio, sr)
  features.append(np.mean(spectral_bandwidth))
  features.append(np.var(spectral_bandwidth))

  #calcul rolloff
  rolloff = librosa.feature.spectral_rolloff(audio, sr)
  features.append(np.mean(rolloff))
  features.append(np.var(rolloff))

  #calcul zero_crossing_rate
  zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
  features.append(np.mean(zero_crossing_rate))
  features.append(np.var(zero_crossing_rate))

  # Calcul du tempo
  onset_env = librosa.onset.onset_strength(audio, sr=sr)
  features.append(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
  # Calcul des moyennes des MFCC
  mfcc = librosa.feature.mfcc(audio)
  for x in mfcc:
    features.append(np.mean(x))
    features.append(np.var(x))

  return features

def getGenra(Path):
  audio, sr= librosa.load(MP32WAV(Path))
  cols = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var','spectral_centroid_mean', 'spectral_centroid_var','spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean','rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var','tempo', 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var','mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean','mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var','mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean','mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var','mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean','mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var','mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean','mfcc20_var']
  items = pd.DataFrame(np.array(pd.DataFrame(audio_pipeline(audio,sr))).T, columns = cols)
  loaded_model = joblib.load('D:/Workspace/tuto/model/model.sav')
  return loaded_model.predict(items)


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        saveFolder = 'D:/Workspace/tuto/static/music/'
        path = request.files['file']
        track = path.filename
        fullPath = saveFolder + track
        path.save(fullPath)
        g = getGenra(fullPath)
        m_ganre = ['blues','classical', 'country','disco', 'hiphop','jazz','metal','pop','reggae','rock']
        return render_template('index.html',g = m_ganre[g[0]], etat = 'details')
    else:
        return "hi"

@app.route('/', methods=['POST'])


@app.route("/details")
def detailsPage():
    return render_template('details.html', g = " ")

@app.route("/")
def hello_world():
    return render_template('index.html', g = " ",etat = "")


if __name__ == "__main__":
    app.run(debug=True)













