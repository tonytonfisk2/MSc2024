import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import Trainer, TrainingArguments

class distillTrainer(Trainer):
    def __init__(self, *args, teacher_model = None, temperature = None, alpha_ce = None, alpha_cos = None, alpha_mlm = None, **kwargs):
        super().__init__(*args,**kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_cos = alpha_cos
        self.alpha_mlm = alpha_mlm
        self.teacher.eval()
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

    def distillation_loss(self, student_outputs, teacher_outputs, attention_mask):
        #soft target probabilities
        s_logits = student_outputs.logits  # (bs, seq_length, voc_size)
        t_logits = teacher_outputs.logits  # (bs, seq_length, voc_size)

        attention_mask = attention_mask.bool()
        mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        
        soft_student = F.log_softmax(s_logits_slct / self.temperature, dim = -1)
        soft_teacher = F.softmax(t_logits_slct / self.temperature, dim = -1)
        #Kullback Leibler Divergence
        distill_loss = self.ce_loss_fct(soft_student, soft_teacher) * (self.temperature**2) 
        return distill_loss

    def cosine_embedding_loss(self, student_outputs, teacher_outputs, attention_mask):
        #cosine embedding loss
        s_hidden_states = student_outputs.hidden_states[-1]  # (bs, seq_length, dim)
        t_hidden_states = teacher_outputs.hidden_states[-1]  # (bs, seq_length, dim)
        
        attention_mask = attention_mask.bool()
        mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
        assert s_hidden_states.size() == t_hidden_states.size()
        dim = s_hidden_states.size(-1)

        s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
        s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
        t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
        t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

        target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
        loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
        return loss_cos

    def compute_loss(self, model, inputs, return_outputs = False):
        
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss
        
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            
        l_ce = self.distillation_loss(student_outputs, teacher_outputs, inputs['attention_mask'])
        loss = self.alpha_ce * l_ce
        
        l_cos = 0
        if self.alpha_cos > 0:
            l_cos = self.cosine_embedding_loss(student_outputs, teacher_outputs, inputs['attention_mask'])
            loss += l_cos * self.alpha_cos
        
        if self.alpha_mlm > 0:
            student_loss = student_loss.mean()
            loss += self.alpha_mlm * student_loss
        
        if(torch.isnan(loss)):
            print('nan loss')
        
        return (loss, student_outputs) if return_outputs else loss

