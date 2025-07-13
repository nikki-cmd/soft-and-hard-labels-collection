import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def getSL(llm, prompt, max_tokens=128000):
    prompt_tokens = llm.tokenize(prompt.encode(), add_bos=True)

    generated_tokens = []
    soft_distributions = []

    llm.eval(prompt_tokens)

    for step in range(max_tokens):
        if not llm.eval_logits:
            raise RuntimeError("Logits not available.")
        logits = llm.eval_logits[-1]

        #probs = softmax(logits)
        probs = logits
        soft_distributions.append(probs)

        next_token = int(np.argmax(probs))
        generated_tokens.append(next_token)

        llm.eval([next_token])

        if next_token == 2:
            print(f"End-of-sequence token generated at step {step+1}. Stopping.")
            break

    generated_text = llm.detokenize(generated_tokens).decode("utf-8", errors="ignore")

    llm.reset()
    return generated_text, soft_distributions
