from llama_cpp import Llama

dataset_path = "datasets/simple_tasks_dataset.csv"
distribution_matrix = "datasets/softlabels_numpy_zip.npz"

model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=256,
    logits_all=True,
    verbose=False
)


