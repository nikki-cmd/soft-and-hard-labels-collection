from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import numpy as np
import configs.configs as cfg

model_path = cfg.model_path

llm = cfg.llm

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def getSL(question):
    prompt = f"Q: {question}\nA:" 
    
    output = llm.create_completion(
        prompt,
        max_tokens=50,
        stop=["Q:", "\n"],
        temperature=0.0,
        logprobs=32000,  # Запрашиваем все логиты
        echo=False
    )
    
    generated_text = output['choices'][0]['text']
    
    print(f"\nQuestion: {question}")
    
    if 'logprobs' in output['choices'][0]:
        tokens = output['choices'][0]['logprobs']['tokens']
        # Получаем все логиты для каждого токена (если доступно)
        all_logits = output['choices'][0]['logprobs']['top_logprobs']
        
        # Создаем матрицу softlabels (n_tokens x vocab_size)
        n_tokens = len(tokens)
        vocab_size = 32000
        softlabels_matrix = np.zeros((n_tokens, vocab_size))
        
        for i, logprobs_dict in enumerate(all_logits):
            # Создаем временный массив для хранения логитов
            logits = np.zeros(vocab_size)
            
            # Заполняем известные логиты из top_logprobs
            for token, logprob in logprobs_dict.items():
                # Нам нужно преобразовать токен в его индекс в словаре
                # Это сложная часть, так как нам нужен доступ к словарю токенов модели
                # Временно используем заполнитель (это нужно доработать)
                token_idx = 0  # Замените на реальный индекс токена
                logits[token_idx] = logprob
            
            # Применяем softmax
            probabilities = softmax(logits)
            softlabels_matrix[i, :] = probabilities
    
    else:
        print("Logprobs not available in the output")
        return generated_text.strip(), None
    
    return generated_text.strip(), softlabels_matrix