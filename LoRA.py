from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

#If only targeting attention blocks of the model
target_modules = ["q_proj", "v_proj"]

#If targeting all linear layers
target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

config = LoraConfig(
    r=16,
    target_modules = target_modules,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")


model = 모델 불러온 거 (llama2)



from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = LlamaForCausalLM.from_pretrained(
    model_path, device_map='auto', load_in_8bit=True,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#########로스 설정##########
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        loss = my_custom_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def my_custom_loss(logits, labels):
    # 여기에 로스 함수 구현
    return loss값


training_args = TrainingArguments(
    output_dir=base_dir,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs = 3.0,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

trainer = MyTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field="text",
    max_seq_length=256,
    args=training_args,
)

#학습(이미지화)
import mlflow
with mlflow.start_run(run_name= ‘run_name_of_choice’):
    trainer.train()

