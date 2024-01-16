# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("../Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("../Llama-2-7b-chat-hf")

# from peft import LoftQConfig, LoraConfig, get_peft_model

# loftq_config = LoftQConfig(loftq_bits=4)           # set 4bit quantization
# lora_config = LoraConfig(..., init_lora_weights="loftq", loftq_config=loftq_config)
# peft_model = get_peft_model(model, lora_config)

import os, torch, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

a = torch.cuda.memory_allocated(0)
print("memory allocated: ", a)

# Dataset
# data_name = "mlabonne/guanaco-llama2-1k"
# training_data = load_dataset("text", data_files={"train": "dataset/train.txt", "test": "dataset/test.txt"}, split="train")
# testing_data = load_dataset("text", data_files={"train": "dataset/train.txt", "test": "dataset/test.txt"}, split="test")

training_data = load_dataset("Samir001/Resume_Summary",split="train").remove_columns("summary_text").rename_column("Resume","text")
#training_data = load_dataset("talanAI/resumesamples", split="train").remove_columns(['Category', 'Email']).rename_column('Resume', 'text')
#training_data = load_dataset("imdb", split="train")

# print("my_data", my_training_data[10])
# print("imdb data", training_data[10])

print(training_data)

# Model and tokenizer names
base_model_name = "../Llama-2-7b-chat-hf"
refined_model = "llama-2-7b-resume-enhanced-2"

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# exit()

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./7b_modified-3",
    num_train_epochs=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=25,
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    # device_map={'': 6}
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    # eval_dataset=testing_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params,

)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuning.train()
# fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)