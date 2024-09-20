import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, TrainerCallback
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

class distillTrainer(Trainer):
    def __init__(self, *args, teacher_model = None, temperature = None, alpha_ce = None, **kwargs):
        super().__init__(*args,**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.layer_groups = [f"transformer.layer.{i}" for i in range(6)] 
        self.current_layer_group = 0
        self.unfrozen_layers = set()
        self.layer_logs = []
        
    def compute_loss(self, model, inputs, return_outputs = False):
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
        #Compare layer by layer, min(H' - H).
        student_contexts = student_outputs.contexts[self.current_layer_group]
        teacher_contexts = teacher_outputs.contexts[self.current_layer_group]
    
        context_loss = F.mse_loss(student_contexts, teacher_contexts)
        
        return (context_loss, student_outputs) if return_outputs else context_loss
        
    def train(self, resume_from_checkpoint=None, **kwargs):
        layer_plots = []
        for layer_group in self.layer_groups:
            print(f"Training layer group: {layer_group}")
            self.switch_to_next_layer_group()
            print(self.get_num_trainable_parameters())
            res = super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
            self.layer_logs.append(self.state.log_history.copy())
            self.current_layer_group += 1
            #self.save_model(f"./results/layer_{layer_group}")
        self.plot_layer_losses()
        return res

    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def switch_to_next_layer_group(self):
        self.freeze_all_layers()
        print("Current layer", self.current_layer_group)
        if self.current_layer_group < len(self.layer_groups):
            current_layer = self.layer_groups[self.current_layer_group]
            for name, param in self.model.named_parameters():
                if current_layer in name and any(qkv in name for qkv in ['q_lin', 'k_lin', 'v_lin']):
                    if name not in self.unfrozen_layers:
                        param.requires_grad = True
                        
            self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
            
            num_training_steps = len(self.train_dataset) // self.args.train_batch_size * self.args.num_train_epochs
            warmup_rate = 0.1  # 10% 
            warmup_steps = int(num_training_steps * warmup_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps, 
                num_training_steps=num_training_steps
            )

    def plot_layer_losses(self):
        fig, axs = plt.subplots(2, 3, figsize=(20, 15))
        for layer, (ax, layer_logs) in enumerate(zip(axs.flatten(), self.layer_logs)):
            train_data = [(log['step'], log['loss']) for log in layer_logs if 'loss' in log]
            eval_data = [(log['step'], log['eval_loss']) for log in layer_logs if 'eval_loss' in log]
            
            if train_data:
                steps, losses = zip(*train_data)
                ax.plot(steps, losses, label='Train Loss')
            if eval_data:
                steps, losses = zip(*eval_data)
                ax.plot(steps, losses, label='Validation Loss')
            
            ax.set_title(f'Layer {layer} Loss')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Loss')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('layer_losses_2.png')
        plt.close()



    