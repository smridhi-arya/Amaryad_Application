import os
from flask import Flask, render_template, request
import time

from sentiment_predict import sentiment_prediction


app = Flask(__name__)
@app.route('/sentiment/<sent>', methods=['GET', 'POST'])
def upload_sentiment(sent):
    #file='आयपीएल पुढे ढकला; महाराष्ट्राच्या चर्चा'
    '''# if request.method == 'POST':
    file = request.form['headline']
        # start=time.time()
        # file1 = open("temp.txt", "wb")
        # file1.write(file)
        # file1.close()
        # file1 = open("temp.txt", "rb")
        # b64=file1.read()
        # file1.close()
        # b64 = file.encode('ascii')

        # print(time.time()-start)
        '''
    print("hey")
    file=sent
    if file:
        print('input received')
        print(file)
        start = time.time()
        print(start)
        extracted_json = sentiment_prediction(file)
        end = time.time()
        print('classify-',end-start)
        print(extracted_json)
        return extracted_json
    else:
        return "Hello"


if __name__ == '__main__':
    print("Server Started......")
    app.run(host="0.0.0.0", port = "8888", threaded=False, debug=True)
    #app.run(host="0.0.0.0",threaded = True, debug=True)
