from llama_cpp import Llama

dataset_path = "datasets/simple_tasks_dataset.csv"
distribution_matrix = "datasets/softlabels_numpy_zip.npz"

model_path = "models/Qwen3-1.7B-Q4_K_M.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    logits_all=True,
    verbose=False
)

start_question = 1

max_new_tokens  = 10
