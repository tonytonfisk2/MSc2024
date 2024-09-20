import comet_ml 
from datasets import load_from_disk
from ta_distiller import distillTrainer
import torch
from transformers import DistilBertConfig, DistilBertForMaskedLM, DataCollatorForLanguageModeling, AutoTokenizer, BertForMaskedLM, EarlyStoppingCallback, Trainer, TrainingArguments
from iDistilbert import iDistilBertForMaskedLM
from sklearn.metrics import accuracy_score


experiment = comet_ml.get_global_experiment()
comet_ml.init(project_name="distilbert_wikipedia")
    
def main():
    #Load dataset
    tokenized_dataset = load_from_disk('/mnt/tony/MSc2024/data/tokenized_wikipedia_20231101.hf') #20231101
    num_samples = len(tokenized_dataset) - 2000000
    train_dataset = tokenized_dataset.select(range(num_samples))
    #eval_dataset = tokenized_dataset["validation"]
    
    #Load tokenizer
    tok_id = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    #Load Models
    teacher_id = "google-bert/bert-base-uncased" 
    teacher_model = BertForMaskedLM.from_pretrained(
            teacher_id,
            output_hidden_states = True,
        )
    
    student_config = DistilBertConfig(
        distance_metric = "manhattan_distance",
        activation_function = "relu",
        signed_inhibitor =  True,
        alpha = 0,
        center = True,
        output_hidden_states = True,
        )
    
    student_model = iDistilBertForMaskedLM(student_config)
    
    initialized_weights = torch.load('distilbert_init/models/q_k_v_layerwise_3.pth')
    student_model.load_state_dict(initialized_weights, strict=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher_model.to(device)
    student_model.to(device)
    EPOCHS = 3
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir = './results',
        num_train_epochs = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        logging_dir = './logs',
        save_strategy="steps",
        logging_steps = 1290,
        save_steps=2582,
        save_total_limit=10,
        seed = 42,
        report_to=['comet_ml', 'tensorboard'],
        warmup_ratio=0.05,
        weight_decay = 0.01,
        gradient_accumulation_steps=64, # batchsize * gradient_steps
        fp16=True,
        lr_scheduler_type="linear",
        max_grad_norm = 5.0,
    )
    
    trainer = distillTrainer(
        teacher_model=teacher_model,
        model=student_model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        #eval_dataset=eval_dataset,
        temperature = 2,
        alpha_ce = 5,
        alpha_cos = 2,
        alpha_mlm = 1,
        tokenizer = tokenizer,
        data_collator = data_collator,
    )
    
    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model('./models')

if __name__ == "__main__":
    main()