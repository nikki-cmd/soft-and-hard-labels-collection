from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import numpy as np

model_path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    logits_all=True,  # Это критически важно
    verbose=False
)

def getHL_with_logits(question):
    prompt = f"Q: {question}\nA:" 
    
    # Используем create_completion вместо __call__
    output = llm.create_completion(
        prompt,
        max_tokens=50,
        stop=["Q:", "\n"],
        temperature=0.0,
        logprobs=100,  # Получаем топ-5 логитов для каждого токена
        echo=False,
    )
    
    # Получаем сгенерированный текст
    generated_text = output['choices'][0]['text']
    
    print(f"\nQuestion: {question}")
    print("Generated tokens with logits distributions:")
    
    # Получаем информацию о токенах и их логитах
    if 'logprobs' in output['choices'][0]:
        tokens = output['choices'][0]['logprobs']['tokens']
        top_logprobs = output['choices'][0]['logprobs']['top_logprobs']
        
        for i, (token, logprobs) in enumerate(zip(tokens, top_logprobs)):
            print(f"\nToken {i+1}: '{token}'")
            for tok, logprob in logprobs.items():
                print(f"  '{tok}': {logprob:.4f}")
    else:
        print("Logprobs not available in the output")
    
    return generated_text.strip()

# Пример использования
questions = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms."
]

for q in questions:
    answer = getHL_with_logits(q)
    print(f"\nFinal answer: {answer}\n{'-'*50}")