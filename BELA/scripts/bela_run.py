import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from .bela.transforms.spm_transform import SPMTransform
from .bela.evaluation.model_eval import ModelEval, load_file

embeddings_path = "./models/embeddings.pt"
ent_catalogue_idx_path = "./models/index.txt"
checkpoint_path = "./models/model_wiki.ckpt"

md_threshold = 0.2
el_threshold = 0.4

# joint_el_mel_new xlm-roberta-large
print(f"Loading model from checkpoint {checkpoint_path}")
model_eval = ModelEval(checkpoint_path, config_name="joint_el_mel_new", embeddings_path=embeddings_path,
                       ent_catalogue_idx_path=ent_catalogue_idx_path)

print(model_eval.process_batch(["Jobs was CEO of Apple"]))

#output = [{'offsets': [9, 16], 'lengths': [3, 5], 'entities': ['Q484876', 'Q312'], 'md_scores': [0.24852867424488068, 0.7043067216873169], 'el_scores': [0.48497316241264343, 0.9504457712173462]}]


#import torch

# Check if GPU is available
#if torch.cuda.is_available():
#    print("GPU is available")
#else:
#    print("GPU is NOT available")
