# model_tools.py
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)

def getHL(question):
    llm = Llama(model_path=model_path, logits_all=False, verbose=False)
    
    prompt = f"Q: {question}\nA:" 
    
    response = llm(prompt, max_tokens=50, stop=["Q:", "\n"])
    raw_text = response['choices'][0]['text']
    
    answer = raw_text.strip()
    
    return answer


