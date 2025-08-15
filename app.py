import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
# Load the pre-trained model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_prediction_model.keras')

## Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

#function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
  token_list=tokenizer.texts_to_sequences([text])[0]
  if(len(token_list)<max_sequence_len):
    token_list=token_list[-(max_sequence_len-1):]##ensures the sequence length matches max_sequence
  
  token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predicted_word_index= np.argmax(predicted, axis=1)
  
  for(word,index) in tokenizer.word_index.items():
    if(index==predicted_word_index):
      return word
  return None
  

##streamlit app
st.title("Next Word Prediction")
# Input field for user text
input_text = st.text_input("Enter a sentence:", "to be or not to be")
if st.button("Predict Next Word"):
  max_sequence_len = model.input_shape[1] + 1
  next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
  st.write(f"The next word: {next_word if next_word else 'No prediction available'}")

