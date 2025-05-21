!pip install gradio
import gradio as gr
import numpy as np
import pickle

scaler = pickle.load(open('scaler.pkl','rb'))
model = pickle.load(open('regresion.pkl','rb'))

def calculate_gold_rate(usd_inr):
  scaled_input = scaler.transform(np.array(usd_inr).reshape(1,-1))
  return round(model.predict(scaled_input[0][0],2))
demo = gr.Interface(
    fn=calculate_gold_rate,
    inputs=["number"],
    outputs=["number"],
    title = 'How much is one gram gold now'
)

demo.launch()