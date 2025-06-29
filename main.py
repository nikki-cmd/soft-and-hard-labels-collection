from configs import configs
from data_tools.dataloader import Loader 
from data_tools.uploader import SoftsUploader, AnswersUploader
from tools.get_soft_labels import getSL 

loader = Loader(path=configs.dataset_path)
loaded_dataset = loader.get_questions()

questions = loaded_dataset['question']

answers = []
Slabels = []

for q in questions:
    q_id = loaded_dataset.loc[loaded_dataset['question'] == q, 'id'].values[0]
    print(f"Processing question#{q_id}:", q)
    
    answer, softlabels = getSL("Q:"+q+"\nA:")
    print("Answer:", answer)
    
    answers.append(answer)
    Slabels.append(softlabels)
    uploader = SoftsUploader(distributions_matrix=Slabels, question_id=q_id)
    uploader.upload_distributions()

print("Saving into files")
   
uploader = AnswersUploader(path=configs.dataset_path, answers=answers)
uploader.upload_answers()
