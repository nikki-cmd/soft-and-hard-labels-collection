from configs import configs
from data_tools.dataloader import Loader 
from data_tools.uploader import Uploader
from tools.get_soft_labels import getSL 

loader = Loader(path=configs.dataset_path)
questions = loader.get_questions()

answers = []
Slabels = []

for q in questions:
    print("Processing question")
    
    answer, softlabels = getSL(q)
    
    answers.append(answer)
    Slabels.append(softlabels)

print("Saving into files")
        
uploader = Uploader(path=configs.dataset_path, answers=answers, distributions_matrix=Slabels)
uploader.upload_answers()
uploader.upload_distributions()