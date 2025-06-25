from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import numpy as np
import configs.configs as cfg

model_path = cfg.model_path

llm = cfg.llm

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def getSL(prompt, max_tokens=10):
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

    softlabels_array = np.array(soft_distributions)
    np.save("softlabels.npy", softlabels_array)
    print(f"Soft labels saved to softlabels.npy, shape: {softlabels_array.shape}")
    llm.reset()
    return generated_text, soft_distributions
