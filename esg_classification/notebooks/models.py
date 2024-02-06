# ------------------------------------------------------------------------
# ---------------------------- IMPORTS -----------------------------------
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
# ------------------------------------------------------------------------
# ---------------------------- MODEL CB  ---------------------------------
# ------------------------------------------------------------------------

class ModelCB(nn.Module):
  def __init__(self,checkpoint,num_labels,id2label): 
    super(ModelCB,self).__init__() 
    self.num_labels = num_labels 
    self.id2label = id2label

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(768,num_labels) # load and initialize weights

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)


# ------------------------------------------------------------------------
# ------------------------ MODEL CB - LONG -------------------------------
# ------------------------------------------------------------------------
class ModelCBLong(nn.Module):
  def __init__(self,checkpoint,num_labels, id2label, max_length = 1024): 
    super(ModelCBLong,self).__init__() 
    self.num_labels = num_labels 
    self.id2label = id2label

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(768,num_labels) # load and initialize weights
    
    self.model.config.max_position_embeddings = max_length
    self.model.base_model.embeddings.position_ids = torch.arange(max_length).expand((1, -1)).to(torch.long)
    self.model.base_model.embeddings.token_type_ids = torch.zeros(max_length).expand((1, -1)).to(torch.long)  
    
    orig_pos_emb = self.model.base_model.embeddings.position_embeddings.weight
    self.model.base_model.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))

  def forward(self, input_ids=None, attention_mask=None,labels=None):
    #Extract outputs from the body
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

    #Add custom layers
    sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

    logits = self.classifier(sequence_output[:,0,:].view(-1,768)) # calculate losses
    
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,attentions=outputs.attentions)




# ------------------------------------------------------------------------
# ---------------------------- MODEL CBL ---------------------------------
# ------------------------------------------------------------------------

class ModelCBL(nn.Module):
  def __init__(self,checkpoint,num_labels, id2label): 
    super(ModelCBL,self).__init__() 
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



# ------------------------------------------------------------------------
# ------------------------ MODEL CBL - LONG ------------------------------
# ------------------------------------------------------------------------
class ModelCBLlong(nn.Module):
  def __init__(self,checkpoint,num_labels, id2label, max_length = 512): 
    super(ModelCBLlong,self).__init__() 
    self.num_labels = num_labels 
    self.id2label = id2label

    #Load Model with given checkpoint and extract its body
    self.model = AutoModel.from_pretrained(
        checkpoint,
        config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True)
        )
    self.dropout = nn.Dropout(0.1) 
    self.classifier = nn.Linear(1024,num_labels) # load and initialize weights
    
    self.model.config.max_position_embeddings = max_length
    self.model.base_model.embeddings.position_ids = torch.arange(max_length).expand((1, -1)).to(torch.long)
    self.model.base_model.embeddings.token_type_ids = torch.zeros(max_length).expand((1, -1)).to(torch.long) 
    orig_pos_emb = self.model.base_model.embeddings.position_embeddings.weight
    self.model.base_model.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))

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