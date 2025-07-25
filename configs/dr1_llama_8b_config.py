from llama_cpp import Llama

dataset_path = "datasets/s1K-1.1-prepared.csv"
distribution_matrix = "datasets/softlabels_numpy_zip.npz"

model_path = "models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"

llm = Llama(model_path=model_path, n_ctx=700, n_threads=4, logits_all=True,
    verbose=False)

start_question = 1

max_new_tokens  = 10
