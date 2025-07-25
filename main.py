from configs import configs
from data_tools.dataloader import Loader 
from data_tools.uploader import SoftsUploader, AnswersUploader
from tools.get_soft_labels import getSL 
from tools import parser as parse
import logging
from datetime import datetime
import time
import sys
import argparse
import importlib

parser = argparse.ArgumentParser(description='Run soft labels collection')
parser.add_argument('--config', type=str, required=True, 
                        help='Config module name (e.g. "llama_config" from configs/)')
args, unknown = parser.parse_known_args()

extra_args = parse.parse_key_value_args(unknown)

config = importlib.import_module(f'configs.{args.config}')

for key, value in extra_args.items():
    if hasattr(config, key):
        setattr(config, key, value)
        
if not hasattr(config, 'start_question'):
    config.start_question = 1
    
if not hasattr(config, 'max_new_tokens'):
    config.max_new_tokens = 10
    
if not hasattr(config, 'temperature'):
    config.temperature = 1

current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"runs/logs/run_{current_time}.log", encoding='utf-8'),
        logging.StreamHandler(stream=open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False))
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Start from question #{config.start_question}")
loader = Loader(path=config.dataset_path, start_question=config.start_question)
loaded_dataset = loader.get_questions()
questions = loaded_dataset['question']

answers = []

for idx, q in enumerate(questions):
    start_time = time.time()
        
    q_id = loaded_dataset.iloc[idx]['id']
        
    logger.info(f'Processing question #{q_id}')
    
    try:
        answer, softlabels = getSL(config.llm, q, max_new_tokens=config.max_new_tokens, temperature=config.temperature)
        processing_time = time.time() - start_time
        logger.info(f"[OK] Answer generated (took {processing_time:.2f}s)")
            
        answers.append(answer)
            
        uploader = SoftsUploader(distributions_matrix=softlabels, question_id=q_id, current_time=current_time)
        uploader.upload_distributions()
            
        logger.info(f"Uploaded distributions for question #{q_id}")
    except Exception as e:
        logger.error(f"[ERROR] Error processing question ID {q_id}: {str(e)}", exc_info=True)
        continue
    
logger.info("Uploading answers...")
uploader = AnswersUploader(path=config.dataset_path, answers=answers, questions=questions, dataset_name=current_time)
uploader.upload_answers()
logger.info("Uploaded successfully.")
    