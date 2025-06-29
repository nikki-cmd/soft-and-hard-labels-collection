from configs import configs
from data_tools.dataloader import Loader 
from data_tools.uploader import SoftsUploader, AnswersUploader
from tools.get_soft_labels import getSL 
import logging
from datetime import datetime
import time
import sys

current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/run_{current_time}.log", encoding='utf-8'),
        logging.StreamHandler(stream=open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False))
    ]
)
logger = logging.getLogger(__name__)

loader = Loader(path=configs.dataset_path)
loaded_dataset = loader.get_questions()

questions = loaded_dataset['question']

answers = []
Slabels = []

for q in questions:
    start_time = time.time()
    
    q_id = loaded_dataset.loc[loaded_dataset['question'] == q, 'id'].values[0]
    
    logger.info(f'Processing question #{q_id}:{q}')
    
    try:
        answer, softlabels = getSL("Q:"+q+"\nA:")
        processing_time = time.time() - start_time
        logger.info(f"✅ Answer generated (took {processing_time:.2f}s): {answer}")
        
        answers.append(answer)
        Slabels.append(softlabels)
        
        uploader = SoftsUploader(distributions_matrix=Slabels, question_id=q_id, current_time=current_time)
        uploader.upload_distributions()
        
        logger.info(f"Uploaded distributions for question #: {q_id}")
    except Exception as e:
        logger.error(f"❌ Error processing question ID {q_id}: {str(e)}", exc_info=True)
        continue
   
logger.info("Uploading answers...")
uploader = AnswersUploader(path=configs.dataset_path, answers=answers)
uploader.upload_answers()
logger.info("Uploaded successfully.")
