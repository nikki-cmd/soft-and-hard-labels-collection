from config import Config
from utils import read_dataset, store_hard_soft_labels_on_step
from model import Model

import logging
import time


if __name__ == '__main__':
    config = Config.from_args()

    logger = logging.getLogger(__name__)
    logger.info(f"Run script: {config.current_time}")

    logger.info("Config parameters:")
    for k, v in config.as_dict().items():
        logger.info(f"{k}: {v}")

    dataset = read_dataset(config)
    model = Model(config)

    for _, row in dataset.iterrows():
        q_id, question = row["id"], row["question"]
        logger.info(f"Processing question #{q_id}")

        try:
            start_time = time.time()
            hard_labels_text, soft_labels_probs, soft_labels_indices = model.process(question)
            processing_time = time.time() - start_time
            logger.info(f"[OK] Answer generated (took {processing_time:.2f}s)")

            store_hard_soft_labels_on_step(
                q_id,
                hard_labels_text,
                soft_labels_probs,
                soft_labels_indices,
                config
            )

            logger.info(f"[OK] Distributions uploaded, question #{q_id} done")
        except Exception as e:
            logger.error(f"[ERROR] Error processing question ID {q_id}: {str(e)}", exc_info=True)
            exit(1)
