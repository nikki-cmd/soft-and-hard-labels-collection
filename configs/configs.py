from llama_cpp import Llama

dataset_path = "datasets/s1K-1.1-prepared.csv"
answered_dataset_path = "datasets/answered_dataset.csv"
distribution_matrix = "datasets/softlabels_numpy_zip.npz"

start_question = 1

max_new_tokens  = 10

temperature = 1