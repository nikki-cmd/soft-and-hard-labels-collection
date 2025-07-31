import numpy as np
import sys

def softmax(x, temperature):
    x = np.array(x, dtype=np.float64)
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled))
    return e_x / e_x.sum()

def top_p_get_logits(logits, p=0.9, temperature=1.0):
        logits = np.array(logits, dtype=np.float64)
        probs = softmax(logits, temperature)
        
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        cutoff = int(np.sum(cumulative_probs < p)) + 1
        top_p_indices = sorted_indices[:cutoff]
        
        top_p_logits = logits[top_p_indices]

        top_p_probs = probs[top_p_indices]
        top_p_probs = softmax(logits[top_p_indices])
        
        selected_index = np.random.choice(top_p_indices, p=top_p_probs)
        
        return selected_index, top_p_indices, top_p_logits

def getSL(llm, prompt, max_new_tokens=10, top_p=0.9, temperature=0.8):
    prompt_tokens = llm.tokenize(prompt.encode(), add_bos=True)
    
    llm.reset()
    llm.eval(prompt_tokens)
    
    hard_labels_text = ""
    soft_labels_logits_list = []

    for step in range(max_new_tokens):
        if not llm.eval_logits:
            raise RuntimeError("Logits not available.")
        logits = llm.eval_logits[-1]
        
        try:
            next_token, top_p_indices, top_p_logits = top_p_get_logits(
                logits, p=top_p, temperature=temperature
            )
        except Exception as e:
            print(f"Ошибка на шаге {step}: {e}")
            next_token = int(np.argmax(logits))  # fallback к greedy
            top_p_indices = np.array([next_token])
            top_p_logits = np.array([logits[next_token]])

        token_text = llm.detokenize([next_token]).decode('utf-8', errors='ignore')
        hard_labels_text += token_text

        soft_labels_logits_list.append(top_p_logits)

        llm.eval([next_token])
        
        progress = (step + 1) / max_new_tokens
        filled = int(round(20 * progress))
        bar = '=' * filled + '-' * (20 - filled)
        percent = int(round(100 * progress))
        
        sys.stdout.write(f"\r[{bar}] {percent}% ({step+1}/{max_new_tokens} tokens)")
        sys.stdout.flush()
    
    return hard_labels_text, soft_labels_logits_list
