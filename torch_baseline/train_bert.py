# coding:utf-8

import tokenizers
from transformers import BertTokenizer, LineByLineTextDataset,\
    BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling,\
    Trainer, TrainingArguments


filepath = '../data/train_data1.txt'  # your dataset:train.tsv + test.tsv

bwpt = tokenizers.BertWordPieceTokenizer(vocab_file=None)
bwpt.train(
    files=[filepath],
    vocab_size=20708,
    min_frequency=1,
    limit_alphabet=1000
)

bwpt.save('./model')


vocab_file_dir = './model/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=filepath,
    block_size=64
)

config = BertConfig(
    vocab_size=20708,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)

model = BertForMaskedLM(config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model('./model')