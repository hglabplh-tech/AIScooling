import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
messages = [
    {"role": "system", "content": "You are a helpful and brief AI assistant."},
    {"role": "user", "content": "Explain quantum computing in one sentence."}
]
model_id = "meta-llama/Llama-3.2-1B-Instruct"
def generate_answer(prompt):

    # 2. Load tokenizer and model from pre-trained
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"  # Automatically allocates to GPU if available
        )


    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print("AI:", response)
    return response
