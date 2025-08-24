from config import Config
from llama_cpp import Llama
import numpy as np
import logging


logger = logging.getLogger(__name__)


class Model:
    def __init__(self, config: Config):
        self._config = config

        self._llm = Llama(
            model_path=str(config.model_path),
            logits_all=True,
            verbose=False,
            n_ctx=2423 + config.max_new_tokens,  # 2423 -- upper bound, len of the longest question
            n_gpu_layers=-1 if config.cuda else 0,
            # n_threads=4,
        )

    def process(self, question: str) -> tuple[str, dict[int, list[np.ndarray]], list[np.ndarray]]:
        prompt_tokens = self._llm.tokenize(question.encode(), add_bos=True)

        self._llm.reset()
        self._llm.eval(prompt_tokens)

        hard_labels_text = ""
        soft_labels_probs = {T: [] for T in self._config.temperatures}
        soft_labels_indices = []

        for step in range(self._config.max_new_tokens):
            if not self._llm.eval_logits:
                raise RuntimeError("Logits not available.")
            logits = self._llm.eval_logits[-1]
            logits = np.array(logits)

            next_token_ind, top_p_indices, probs_to_store = self._top_p_get_logits(logits)
            soft_labels_indices.append(top_p_indices)
            for T in soft_labels_probs.keys():
                soft_labels_probs[T].append(probs_to_store[T])
            token_text = self._llm.detokenize([next_token_ind]).decode('utf-8', errors='ignore')
            hard_labels_text += token_text

            # stop if "end of sequence"
            if next_token_ind == self._llm.token_eos():
                break

            self._llm.eval([next_token_ind])

            if (step + 1) % (self._config.max_new_tokens // 10) == 0:
                logger.info(f"Step {step + 1}/{self._config.max_new_tokens}")

        logger.info(f"Done in {len(soft_labels_indices)} steps")
        return hard_labels_text, soft_labels_probs, soft_labels_indices

    def _top_p_get_logits(self, logits: np.ndarray) -> tuple[int, np.ndarray, dict[int, np.ndarray]]:
        # probs @ T=1
        probs = self._softmax(logits)

        # find top-p tokens
        sorted_indices = np.argsort(probs)[::-1]
        cumulative_probs = np.cumsum(probs[sorted_indices])
        cutoff = int(np.sum(cumulative_probs < self._config.top_p)) + 1
        top_p_indices = sorted_indices[:cutoff]

        # sample next token
        top_p_probs = self._softmax(logits[top_p_indices])
        next_token_ind = np.random.choice(top_p_indices, p=top_p_probs)

        # store probs @ T=1,2,3,5
        probs_to_store = dict()
        for T in self._config.temperatures:
            probs_to_store[T] = self._softmax(logits, T)[top_p_indices]

        return next_token_ind, top_p_indices, probs_to_store

    def _softmax(self, x: np.ndarray, temperature: float = 1.) -> np.ndarray:
        x_scaled = x / temperature
        e_x = np.exp(x_scaled - np.max(x_scaled))
        return e_x / e_x.sum()
