from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM
from config.config import cfg
from model.model import model
from dataset.dataset import compute_metrics,data_collator
#SET CUDA AVAILABLE DEVICES
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['model']['CUDA_VISIBLE_DEVICES']
if cfg['model']['train_loop_type'] == "huggingface":
    training_args = TrainingArguments(
        output_dir=cfg['model']['output_dir'],
        num_train_epochs=cfg['model']['num_train_epochs'],
        per_device_train_batch_size=cfg['model']['per_device_train_batch_size'],
        per_device_eval_batch_size=cfg['model']['per_device_eval_batch_size'],
        warmup_steps=cfg['model']['warmup_steps'],
        weight_decay=cfg['model']['weight_decay'],
        logging_dir=cfg['model']['logging_dir'],
        logging_steps=cfg['model']['logging_steps'],
        save_steps=cfg['model']['save_steps'],
        evaluation_strategy=cfg['model']['evaluation_strategy'],
        eval_steps=cfg['model']['eval_steps'],
        save_total_limit=cfg['model']['save_total_limit'],
        load_best_model_at_end=cfg['model']['load_best_model_at_end'],
        metric_for_best_model=cfg['model']['metric_for_best_model'],
        greater_is_better=cfg['model']['greater_is_better'],
        report_to=cfg['model']['report_to'],
        run_name=cfg['model']['run_name'],
        seed=cfg['model']['seed'],
        disable_tqdm=cfg['model']['disable_tqdm']
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=cfg['model']['train_dataset'],
        eval_dataset=cfg['model']['eval_dataset'],
        compute_metrics=compute_metrics
    )
else:
    #add text of to be implemented on Error
    raise NotImplementedError("Only huggingface train loop is implemented for now.")