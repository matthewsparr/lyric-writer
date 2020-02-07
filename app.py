"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/

This file creates your application.
"""

import os
from flask import Flask, render_template, request, redirect, url_for
import pickle

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['KERAS_BACKEND'] = 'theano'
import keras
#import tensorflow as tf
#import keras
from keras.models import model_from_json
import numpy as np
import nltk




app = Flask(__name__, static_url_path='/static')

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'this_should_be_configured')

artists = pickle.load(open('artist_list.pkl', 'rb'))
text_tokens = pickle.load(open('text_tokens.pkl', 'rb'))
maxlen = 6

#cfg = tf.ConfigProto(allow_soft_placement=True)
#cfg.gpu_options.allow_growth = True


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('artist_tokenizer.pickle', 'rb') as handle:
    artist_tokenizer = pickle.load(handle)

keys = []
for key,value in artist_tokenizer.word_index.items():
    keys.append(key)

#keras.backend.clear_session()

json_file = open('lyric_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
global model
model = model_from_json(loaded_model_json)
model.load_weights("lyric_model.h5")


json_file = open('artist_model.json', 'r')
loaded_model_json_ = json_file.read()
json_file.close()
global artist_model
artist_model = model_from_json(loaded_model_json_)
artist_model.load_weights("artist_model.h5")

#global graph
#graph = tf.get_default_graph()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_from_seed(seed_word, sample_flag, temp, length):
    generated = seed_word
    possible_starters = []
    for index,value in enumerate(text_tokens):
        if value==seed_word:
            possible_starters.append(index)
    chosen_index = np.random.choice(possible_starters)
    sentence = text_tokens[chosen_index:chosen_index+maxlen]
    for i in sentence:
        generated+=" " + i
    for i in range(length):
        x_pred = np.array(tokenizer.texts_to_sequences(sentence)).reshape(1,maxlen)
        if sample_flag:
            #with graph.as_default():
            preds = model.predict(x_pred)[0]
            pred = sample(preds, temp)
        else:
            #with graph.as_default():
            pred = model.predict_classes(x_pred)[0]
        next_word = tokenizer.sequences_to_texts([[pred]])[0]
        sentence.append(next_word)
        sentence = sentence[1:]
        generated += " " + next_word
    return generated

def generate_lyrics_(seed_word, sample_flag, temp):
    total_words_needed = 100
    num_artists = 20

    lyric_text = generate_from_seed(seed_word, sample_flag, temp, total_words_needed)
    lyric_text = nltk.sent_tokenize(lyric_text)
    chorus = lyric_text[:4]
    song_title = np.random.choice(chorus)[:-1].title()
    #song_title = chorus[1][:-1].title()
    verse1 = lyric_text[5:9]

    next_lyric_text = generate_from_seed(verse1[0].split()[0], sample_flag, temp, total_words_needed)
    next_lyric_text = nltk.sent_tokenize(next_lyric_text)
    verse2 = next_lyric_text[0:4]
    
    final_lyric_text = generate_from_seed(seed_word, sample_flag, temp, total_words_needed)
    final_lyric_text = nltk.sent_tokenize(final_lyric_text)
    bridge = final_lyric_text[0:6] 
    
    lyrics = ""
    for lines,section_name in zip([verse1, chorus, verse2, chorus, bridge, chorus], 
                              ['verse1','chorus','verse2','chorus','bridge','chorus']):
        lyrics += "[ " + section_name + " ]" + '<br>'
        for i in lines:
            lyrics += " " + i[:-1]
            lyrics += '<br>'
        lyrics += '<br>' + '<br>'
        
    
    artist_probs = np.zeros(num_artists)
    line_count = 0
    for lines in [verse1, chorus, verse2, chorus, bridge, chorus]:
        for i in lines:
            line_count += 1
            artist,probs = predict_artist(i[:-1])
            artist_probs = artist_probs + probs
    artist_probs = artist_probs/line_count
    print("Inspired by:")
    top_artists = artist_probs.argsort()[-3:][::-1]
    artist_results = []
    for i in top_artists:
        artist_results.append((artists[i],np.round(artist_probs[i]*100)))

    return lyrics, artist_results, song_title

def predict_artist(lyric_line):
    lyric_line = lyric_line.split()
    data = []
    for key in keys:
        if key in lyric_line:
            data.append(1) 
        else:
            data.append(0)
    data = np.array(data).reshape(1,-1)
    #with graph.as_default():
    preds = artist_model.predict(data)
    probs = artist_model.predict_proba(data)[0]
    return artists[np.argmax(preds)], probs


###
# Routing for your application.
###

@app.route('/')
def home():
    """Render website's home page."""
    return render_template('home.html', lyrics="", artist_names=[" ", " ", " "], artist_percents=[" ", " ", " "],
                            show_images=False, error=False)


@app.route('/generate_lyrics',methods=['POST']) 
def generate_lyrics():
    seed_word = str(request.form['seed_word']).split()[0].lower()
    originality = (float(request.form['originality'])+1)/100
    error = False
    show_images = True
    try:
        lyrics, artist_results, song_title = generate_lyrics_(seed_word=seed_word, sample_flag=True, temp=originality)

        artist_names, artist_percents = [],[]
        for i in artist_results:
            artist_names.append(i[0])
            artist_percents.append(i[1])
        artist_images = []
        for artist in artist_names:
            #artist_images.append(artist+'.jpg')
            artist_images.append("/static/img/" + artist + ".jpg")
        print(artist_images[0]) 
    except:
        error = True
        show_images = False
        lyrics = artist_names = artist_percents = artist_images = song_title = " "
    return render_template('home.html', lyrics=lyrics, artist_names=artist_names, error=error,
                                        artist_percents=artist_percents, show_images=show_images, artist_images=artist_images,
                                        song_title=song_title)


if __name__ == '__main__':
    app.run(debug=True)






