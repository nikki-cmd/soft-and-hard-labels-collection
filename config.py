from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import argparse
import logging
import sys


DATASET_FILENAME = "s1K-1.1-prepared.csv"

config_to_model_mapping = {
    "dr1_llama_8b_q4km": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    "qwen3_1.7b_q4km": "Qwen3-1.7B-Q4_K_M.gguf",
    "dr1_qwen_1.5b_q4km": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    "llama_1.1b_q4km": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
}

DEFAULT_CONFIG = "dr1_llama_8b_q4km"
DEFAULT_MODELS_DIR = "./models"
DEFAULT_DATASETS_DIR = "./datasets"
DEFAULT_LOGS_DIR = "./logs"
DEFAULT_TEMPERATURES = [1, 2, 3, 5]
MAX_NUMBER_OF_NEW_TOKENS = 100_000
NUMBER_OF_QUESTIONS = 1000


@dataclass
class Config:
    model_path: Path
    dataset_path: Path
    logs_path: Path | None = None
    results_dir: Path = field(init=False)

    start_question: int = 1
    max_new_tokens: int = MAX_NUMBER_OF_NEW_TOKENS
    top_p: float = 0.9
    temperatures: list = field(init=False)

    cuda: bool = False

    current_time: str = datetime.now().strftime("%Y_%m_%d_%H_%M")

    def __post_init__(self):
        self.results_dir = self.dataset_path.parent / f"results_{Config.current_time}"
        self.results_dir.mkdir(exist_ok=True)
        self.temperatures = DEFAULT_TEMPERATURES

    def as_dict(self):
        return asdict(self)

    @staticmethod
    def setup_logger(logs_dir: Path | str) -> Path:
        if isinstance(logs_dir, str):
            logs_dir = Path(logs_dir)
        logs_dir.mkdir(exist_ok=True)
        log_path = logs_dir / f"run_{Config.current_time}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler(stream=open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False))
            ]
        )
        return log_path

    @staticmethod
    def from_args():
        parser = argparse.ArgumentParser(description="Hard & Soft labels collection")
        parser.add_argument(
            "-cfg", "--config", type=str, help="Config name",
            choices=list(config_to_model_mapping.keys()), default=DEFAULT_CONFIG
        )
        parser.add_argument(
            "-md", "--models_dir", type=str, help="Models directory",
            default=DEFAULT_MODELS_DIR
        )
        parser.add_argument(
            "-dd", "--datasets_dir", type=str, help="Datasets directory",
            default=DEFAULT_DATASETS_DIR
        )
        parser.add_argument(
            "-ld", "--logs_dir", type=str, help="Logs directory",
            default=DEFAULT_LOGS_DIR
        )
        parser.add_argument(
            "-sq", "--start_question", type=int, help="The number of a question to start with",
            default=Config.start_question
        )
        parser.add_argument(
            "--cuda", action='store_true', help="Enable CUDA support",
            default=Config.cuda
        )
        parser.add_argument(
            "-tp", "--top_p", type=bool, help="Top-p sampling threshold to gather tokens",
            default=Config.top_p
        )
        parser.add_argument(
            "-mnt", "--max_new_tokens", type=int, help="Maximum number of new generated tokens",
            default=Config.max_new_tokens
        )
        args = parser.parse_args()

        if args.config not in config_to_model_mapping:
            raise ValueError(f"Config '{args.config}' not supported.")

        model_path = Path(args.models_dir) / config_to_model_mapping[args.config]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        dataset_path = Path(args.datasets_dir) / DATASET_FILENAME
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if not (1 <= args.start_question <= NUMBER_OF_QUESTIONS):
            raise ValueError(f"start_question must be within 1-{NUMBER_OF_QUESTIONS}, given {args.start_question}")

        if not (0 < args.top_p <= 1):
            raise ValueError(f"top_p must be within 0-1, given {args.top_p}")

        if not (0 < args.max_new_tokens <= MAX_NUMBER_OF_NEW_TOKENS):
            raise ValueError(f"max_new_tokens must be within 1-{MAX_NUMBER_OF_NEW_TOKENS} range, given {args.max_new_tokens}")

        return Config(
            model_path=model_path,
            dataset_path=dataset_path,
            logs_path=Config.setup_logger(args.logs_dir),
            start_question=args.start_question,
            cuda=args.cuda,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
