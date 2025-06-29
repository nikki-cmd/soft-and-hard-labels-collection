from llama_cpp import Llama
from huggingface_hub import hf_hub_download

dataset_path = "datasets/simple_tasks_dataset.csv"
answered_dataset_path = "datasets/answered_dataset.csv"
distribution_matrix = "datasets/softlabels_numpy_zip.npz"

model_path = "models/Qwen3-1.7B-Q4_K_M.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    logits_all=True,
    verbose=False
)


