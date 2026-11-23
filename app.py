
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import gradio as gr

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


# I per últim, l'interficia per fer servir gradio
demo = gr.Interface(
    resumir,
    gr.Textbox(label="Text", placeholder="EScriu aqui...", lines=10),
    gr.Textbox(label="Summary"),
    title="Text Summarization",
    description="Introdueix un text i obté un resum (el text ha de ser en anglés)."
)

demo.launch()
