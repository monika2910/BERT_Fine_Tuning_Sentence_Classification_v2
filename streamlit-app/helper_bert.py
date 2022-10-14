import re

import numpy as np
import pandas as pd

import tensorflow as tf

import numpy as np

import transformers
from transformers import BertTokenizer ,BertTokenizerFast
from transformers import BertConfig, BertModel
from transformers import BertTokenizer

from transformers import BertForSequenceClassification
from keras.utils.data_utils import pad_sequences

import torch

output_dir = '../model_saved_sentence_classification/'


model_load = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer_load=BertTokenizer.from_pretrained(output_dir)

# Labels in our dataset.
labels = [1,0]
max_length = 64  # Maximum length of input sentence to the model.
batch_size = 1


def check_similarity(setence):
    # Tracking variables
    predictions = []
    logits = []

    all_input_ids_load = prep_data_load_model(setence)
    print(all_input_ids_load[0])

    pad_input_ids_load,attention_masks_load=padding_attention_mask(all_input_ids_load)

    #test_data = BertSemanticDataGenerator(sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False, )

    prediction_inputs_load = torch.tensor(pad_input_ids_load)
    prediction_masks_load = torch.tensor(attention_masks_load)

    outputs = model_load(prediction_inputs_load, attention_mask=prediction_masks_load)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy()

    print(logits)
    # np.array(logits[:2])

    # Store predictions and true labels
    predictions.append(logits)
    pred_flat = np.argmax(logits, axis=1)

    print(pred_flat)

    return pred_flat





def prep_data_load_model(setence):
    all_input_ids_load = []
    # all_input_ids=[]

    tokens = tokenizer_load.tokenize(setence)

    # 0 denotes first sentence
    seg_ids = [0] * len(tokens)



    # input ids are generated for the tokens (one question pair)
    input_ids = tokenizer_load.convert_tokens_to_ids(tokens)

    # input ids are stored in a separate list
    all_input_ids_load.append(input_ids)
    print(all_input_ids_load)
    return all_input_ids_load

def padding_attention_mask(all_input_ids_load):

    # padding
    MAX_LEN = 64
    # Pad our input tokens
    pad_input_ids_load = pad_sequences(all_input_ids_load,
                                       maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks_load = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in pad_input_ids_load:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks_load.append(seq_mask)
    print(attention_masks_load)

    return(pad_input_ids_load,attention_masks_load)





