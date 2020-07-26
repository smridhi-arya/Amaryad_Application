from flask import jsonify
from transformers import BertTokenizer
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
# from collections import defaultdict
import torch
import json

def newsclassify(inp):
    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    head=inp
    # head = "सिद्धार्थ जाधव पुन्हा रोहित शेट्टीच्या चित्रपटात; दिसणार खास भूमिकेत"
    # head = "आयपीएल पुढे ढकला; महाराष्ट्राच्या चर्चा"
    # head = "राज्यात करोनाचे १० रुग्ण, गर्दी टाळा; मुख्यमंत्र्यांचं आवाहन"
    headlines=np.array([head],dtype=object)
    # headlines

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    print('OG: ', headlines[0])

    print('Tokenized: ', tokenizer.tokenize(headlines[0]))

    # print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(headlines[0])))


    model=torch.load('BERTMODEL.dms',map_location='cpu')


    input_ids = []

    for h in headlines:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_head = tokenizer.encode(h, add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            # max_length = 48,          # Truncate all sentences.
                            # return_tensors = 'pt'     # Return pytorch tensors.
                       )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_head)

    # print('Original: ', headlines[0])
    # print('Token IDs:', input_ids[0])


    # print('Max sentence length: ', max([len(sen) for sen in input_ids]))

    MAX_LEN = 64
    # print(np.zeros((1,MAX_LEN)))

    # print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    attention_masks = []

    for sent in input_ids:
        att_mask = [float(token_id > 0) for token_id in sent]
    # float??????
        attention_masks.append(att_mask)

    # print("Attention masks", attention_masks)

    prediction_inputs = torch.tensor(input_ids,dtype=torch.long)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(np.array([2]),dtype=torch.long)

    batch_size=16

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

    model.eval()

    predictions, true_labels = [], []

    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        print("PREDICTION  ", logits)
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    prednum = predictions[0].argmax()
    print(prednum)
    if prednum==0:
        pred='State'
    elif prednum==1:
        pred='Entertainment'
    elif prednum==2:
        pred='Sports'
    else:
        pred='Cannot find'
    '''op = {'output':pred}
    output = json.dumps(op,indent=4)
    print(type(output))'''

    return jsonify([{"output": "%s" % pred}])
