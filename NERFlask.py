# import os
from flask import Flask, render_template, request
import time

from NERpred import NERpred

app = Flask(__name__)
@app.route('/ner/<sen>', methods=['GET', 'POST'])
def upload_page(sen):
    file = sen
    if file :
        print('input received')
        start = time.time()
        extracted_json=NERpred(file)
        end=time.time()
        print('classify-',end-start)    
        print(type(extracted_json))
        return extracted_json
    else:
        return "Hello"


if __name__ == '__main__':
    print("Server Started......")
    #app.run(host="0.0.0.0", threaded=True, debug=True)
    app.run(host="0.0.0.0",port="5001",threaded = True, debug=True)
