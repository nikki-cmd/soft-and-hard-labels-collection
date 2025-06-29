from llama_cpp import Llama
from huggingface_hub import hf_hub_download

dataset_path = "datasets/simple_tasks_dataset.csv"
answered_dataset_path = "datasets/answered_dataset.csv"
distribution_matrix = "datasets/softlabels_numpy_zip.npz"

model_path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    logits_all=True,
    verbose=False
)


