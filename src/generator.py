from transformers import pipeline

def load_generator_model(model_name="google/flan-t5-base", device=0):
    return pipeline("text2text-generation", model=model_name, device=device)

def generate_answer(prompt, model, max_length=512):
    result = model(prompt, max_new_tokens=max_length, do_sample=True, temperature=0.7)
    return result[0]["generated_text"].strip()
