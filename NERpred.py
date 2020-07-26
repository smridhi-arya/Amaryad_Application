# import pandas as pd
import re

from flask import jsonify
import pickle
with open('countvect.pickle', 'rb') as file:
    cv = pickle.load(file)
with open('gnb.pickle', 'rb') as model:
    gnb = pickle.load(model)
with open('pers.pickle', 'rb') as p:
    pers = pickle.load(p)
with open('locs.pickle', 'rb') as l:
    locs= pickle.load(l)
with open('locs.pickle', 'rb') as og:
    orgs= pickle.load(og)

def NERpred(tok):
    # d=defaultdict(list)
    tok = re.sub('[\.\,!)?(-]', '', tok)
    tok = tok.strip()

    PER = []
    for match in re.finditer("[^ ]+((बाई)|(राव)|(ने)|(चंद्र)|(नाथ)|(नी)|(कर))", tok):
        PER.append(match.group())
        tok = re.sub(match.group(), '', tok)
    # print(PER)
    tok = tok.strip()

    for i in pers:
        if re.search(i, tok):
            _ = re.search(i, tok).group()
            PER.append(_)
            tok = re.sub(_, '', tok)
    print(PER)
    tok = tok.strip()

    LOC = []
    for i in locs:
        if re.search(i, tok):
            _ = re.search(i, tok).group()
            LOC.append(_)
            tok = re.sub(_, '', tok)
    print(LOC)
    tok = tok.strip()

    ORG = []
    for i in orgs:
        if re.search(i, tok):
            _ = re.search(i, tok).group()
            ORG.append(_)
            tok = re.sub(_, '', tok)
    print(ORG)
    tok = tok.strip()
    O=[]
    d = {}
    d['PER']=PER
    d['LOC']=LOC
    d['ORG']=ORG
    d['O']=O
    print("D",d)
    # print(jsonify(d))

    tok=re.sub('  ',' ',tok)
    s = tok.split(' ')

    for i in s:
        vect = cv.transform([i]).todense()
        pred = gnb.predict(vect)
        conf = gnb.predict_proba(vect)
        d[pred[0]].append(i)
        print(i, pred ,conf)

    for x in d:
        d[x]=str(d[x])
    print(d)
    return jsonify([d])

# print(NERpred("सवित्रिबाई ठाणे गेली. माधवराव सचिन"))