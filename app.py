from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import streamlit as st

# Carregar el model
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

tokenizer = AutoTokenizer.from_pretrained("AKG2/flan-t5-base-xsum-lora")

model = PeftModel.from_pretrained(base_model, "AKG2/flan-t5-base-xsum-lora")


# Definim la funció per a resumir
def resumir(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(**inputs, max_length=64, num_beams=4, length_penalty=2.0)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary


# I per últim, la interfície per fer servir Streamlit
st.title("Text Summarization")
st.write("Introdueix un text i obté un resum (el text ha de ser en anglès).")

text = st.text_area("Text", placeholder="EScriu aqui...", height=250)

if st.button("Summarize"):
    st.text_area("Summary", value=resumir(text), height=100)