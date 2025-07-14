import numpy as np
import sys

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def getSL(llm, prompt, max_new_tokens=10):
    prompt_tokens = llm.tokenize(prompt.encode(), add_bos=True)

    generated_tokens = []
    soft_distributions = []

    llm.eval(prompt_tokens)

    for step in range(max_new_tokens):
        if not llm.eval_logits:
            raise RuntimeError("Logits not available.")
        logits = llm.eval_logits[-1]

        #probs = softmax(logits)
        probs = logits
        soft_distributions.append(probs)

        next_token = int(np.argmax(probs))
        generated_tokens.append(next_token)

        llm.eval([next_token])
        
        progress = (step + 1) / max_new_tokens
        filled = int(round(20 * progress))
        bar = '=' * filled + '-' * (20 - filled)
        percent = int(round(100 * progress))
        
        sys.stdout.write(f"\r[{bar}] {percent}% ({step+1}/{max_new_tokens} tokens)")
        sys.stdout.flush()

    print("Processing finished.")
    generated_text = llm.detokenize(generated_tokens).decode("utf-8", errors="ignore")

    llm.reset()
    return generated_text, soft_distributions
