import argparse
import time

import spacy
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

import torch_model_utils as utils

nlp = spacy.load('fr_core_news_md')
ID_TO_LABEL = {
    0: 'non-esg',
    1: 'environnemental',
    2: 'social',
    3: 'gouvernance'
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

parser = argparse.ArgumentParser(description='Train a text classification model.')

parser.add_argument('--model_checkpoint', type=str, default='camembert/camembert-large', help='Model checkpoint for transformers.')
parser.add_argument('--dataset_path', type=str, default='./data/esg_fr_classification.csv', help='Path to the dataset.')
parser.add_argument('--output_dir', type=str, default='./model-camembert-large', help='Directory for saving model and outputs.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation.')
parser.add_argument('--grad_steps', type=int, default=7, help='Number of gradient accumulation steps.')
parser.add_argument('--dataset_frac', type=float, default=1, help='Fraction of dataset to use.')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training.')
parser.add_argument('--rounds', type=int, default=5, help='Number of training rounds.')

args = parser.parse_args()



output_directory = args.output_dir
DATASET_PATH = args.dataset_path
MODEL_CHECKPOINT = args.model_checkpoint

OUTPUT_MODEL_STATE_DICT_PATH = output_directory + '/models/state_dict/cb_large_model1_sd.pt'
OUTPUT_FULL_MODEL_PATH = output_directory + '/models/full_model/cb_large_model1.pt'

TB_LOGS_PATH = '~/thesis/Tensorboard_logs/psl_minibatching_model'
IMAGE_SAVE_PATH = output_directory + '/images'

# !!!!!! HYPERPARAMETERS !!!!!!
BATCH_SIZE = args.batch_size
BATCH_GRADIENT_ACCUMULATION = args.grad_steps
NUM_LABELS = len(ID_TO_LABEL)
DATASET_FRAC = args.dataset_frac
NUM_EPOCHS = args.num_epochs
ROUNDS = args.rounds


print("\n\n")
print("Arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("\n\n")

utils.create_directories(output_directory)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
# --------------------------------------------------------
# ----------------------  MODEL  -------------------------
# --------------------------------------------------------
class Model1(nn.Module):
  def __init__(self,checkpoint,num_labels, id2label): 
    super(Model1,self).__init__() 
    self.num_labels = num_labels 
    self.id2label = id2label

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(
        checkpoint,
        config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)
        )
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(1024,num_labels) # load and initialize weights

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,1024)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)



if __name__ == "__main__":
    
    train_df, val_df, test_df = utils.split_esg_dataset(dataset_path=DATASET_PATH,
                                                                            batch_size=BATCH_SIZE,
                                                                            datasize_frac=DATASET_FRAC,
                                                                            return_dataloader=False)
    
    
    print("\n Creating dataloaders...\n")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    trainloader, valoader, testloader = utils.df_converter(train_df, val_df, test_df, output="dataloader", tokenizer=TOKENIZER)
    
    # print("\n loading model...\n")
    

    train_acc = []
    train_f1 = []

    test_acc = []
    test_f1 = []
    
    
    
    max_f1= 0
    start_time = time.time()
    for _ in tqdm(range(ROUNDS)):
        
        model=Model1(checkpoint=MODEL_CHECKPOINT,num_labels=NUM_LABELS,id2label=ID_TO_LABEL).to(device)
        
        
        f1_train, acc_train, epoch_loss_list = utils.train_model(model,
                                                                 trainloader,
                                                                 valoader,
                                                                 num_epochs=NUM_EPOCHS,
                                                                 accumulate_grad_batches=BATCH_GRADIENT_ACCUMULATION,
                                                                 lr_scheduler='cosine')
        train_f1.append(f1_train)
        train_acc.append(acc_train)
        
        f1_test, acc_test = utils.test_model(model, testloader)
        test_acc.append(acc_test)
        test_f1.append(f1_test)
        
        if f1_test > max_f1:
            max_f1 = f1_test
            torch.save(model, OUTPUT_FULL_MODEL_PATH)
            torch.save(model.state_dict(), OUTPUT_MODEL_STATE_DICT_PATH)
            final_epoch_loss_list = epoch_loss_list
        
    end_time = time.time()
    
    # print(f"\n\n with {ROUNDS} rounds, {NUM_EPOCHS} epochs and {DATASET_FRAC*100}% of the test dataset, we get the following results:\n")
    print(f"\n\n Time taken: {end_time-start_time:.2f} seconds")
    print(f"{ROUNDS} rounds, {NUM_EPOCHS} epochs and {DATASET_FRAC*100}% of the dataset ")
    
    print(f"Final epoch loss: {final_epoch_loss_list}")
    utils.plot_epoch_loss(final_epoch_loss_list, save_path=IMAGE_SAVE_PATH+'/epoch_loss.png')
    
    print("Train results:")
    print(f"accuracy: {train_acc}")
    print(f"f1 score: {train_f1}")
    
    utils.plot_score(train_acc, trainloader, score_type="accuracy", save_path=IMAGE_SAVE_PATH+'/train_accuracy.png',dataset_title="train dataset")
    utils.plot_score(train_f1, trainloader, score_type="f1 score", save_path=IMAGE_SAVE_PATH+'/train_f1.png',dataset_title="train dataset")
    
    print("Test results:")
    print(f"accuracy: {test_acc}")
    print(f"f1 score: {test_f1}")
    
    utils.plot_score(test_acc, testloader, score_type="accuracy", save_path=IMAGE_SAVE_PATH+'/test_accuracy.png',dataset_title="test dataset")
    utils.plot_score(test_f1, testloader, score_type="f1 score", save_path=IMAGE_SAVE_PATH+'/test_f1.png',dataset_title="test dataset")
