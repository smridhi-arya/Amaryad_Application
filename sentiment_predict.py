import pickle
import numpy as np
from sklearn.preprocessing import scale
from flask import jsonify
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import TaggedDocument

nltk.download('punkt')
nltk.download('wordnet')

def labelize(phrase, label_type):
    labelized = []
    for i,v in enumerate(phrase):
        label = '%s:%s'%(label_type,i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized

def buildPhraseVector(tokens, size):
    tfidf = pickle.load(open('TFIDF', 'rb'))
    w2v = pickle.load(open('W2V_MODEL', 'rb'))
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

def sentiment_prediction(sentence):
    model = pickle.load(open('SENT_MODEL_65', 'rb'))

    inp=sentence

    inp = word_tokenize(inp)
    punct = ['.', ',', ':', '``', '--', '-', '', '\'s', '\'', '&', '$', '#', '\'\'', '`']

    inp = [w for w in inp if not w in punct]
    lemmatizer = WordNetLemmatizer()

    inp = [lemmatizer.lemmatize(w) for w in inp]
    inp = labelize([inp], 'TEST')
    inp = np.concatenate([buildPhraseVector(z, 100) for z in inp[0].words])  # in map(lambda x: x.words, inp)])
    inp = scale(inp)
    res = model.predict_classes(inp)
    print("done")
    res[res == 2] = -1
    prednum = sum(res)

    if prednum > 0:
        pred = 'Positive'
    elif prednum < 0:
        pred = 'Negative'
    elif prednum == 0:
        pred = 'Neutral'
    else:
        pred = 'Cannot find'
    print(pred)
    return jsonify([{"output": "%s" % pred}])
    #return (pred)


#abc="बागी सिनेमाच्या शृंखलेतील यंदाचा 'बागी ३' हा सिनेमा कथारुपी अत्यंत कमजोर आहे. सिनेमा ऐक्शन एंटरटेनिंग बनवण्यासाठी निर्माते आणि दिग्दर्शकांनी निर्मितीमूल्य जरी वाढवलं असलं तरी सिनेमा मनोरंजन करण्यात कमी पडतो."
#abc = " आमचे गाणे किती सुंदर आहे, "
#print(sentiment_predict(abc))