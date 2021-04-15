# -*- coding: utf-8 -*-
# +
import random
import numpy as np
random.seed(42)
np.random.seed(42)

import json
from sklearn.metrics import roc_auc_score
from bert4keras.backend import keras, K, set_gelu, search_layer
from bert4keras.tokenizers import load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import *
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import truncate_sequences
from tqdm import tqdm
from keras.layers import Dropout, Dense, Lambda


# set_gelu('tanh')

min_count = 5
maxlen = 32
batch_size = 64
config_path = '/share/lichongyang/code/qamatch/oppo-text-match/pretrain_model/nezha2/originmodel/bert_config.json'
checkpoint_path = '/share/lichongyang/code/qamatch/oppo-text-match/pretrain_model/nezha2/originmodel/model.ckpt-900000'
dict_path = '/share/lichongyang/code/qamatch/oppo-text-match/pretrain_model/nezha2/originmodel/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            truncate_sequences(maxlen, -1, a, b)
            D.append((a, b, c))
    return D


# 加载数据集
data = load_data(
    '/share/lichongyang/code/qamatch/oppo-text-match/data/gaiic_track3_round1_train_20210228.tsv'
)
train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]

# train_data_0 = [d for d in data if d[2]==0]
# train_data_1 = [d for d in data if d[2]==1]

test_data = load_data(
    '/share/lichongyang/code/qamatch/oppo-text-match/data/gaiic_track3_round1_testA_20210228.tsv'
)
# 统计词频
tokens = {}
for d in data + test_data:
    for i in d[0] + d[1]:
        tokens[i] = tokens.get(i, 0) + 1

tokens = {i: j for i, j in tokens.items() if j >= min_count}
tokens = sorted(tokens.items(), key=lambda s: -s[1])
tokens = {
    t[0]: i + 7
    for i, t in enumerate(tokens)
}  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes

# BERT词频
counts = json.load(open('counts.json'))
del counts['[CLS]']
del counts['[SEP]']
token_dict = load_vocab(dict_path)
freqs = [
    counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
]
keep_tokens = list(np.argsort(freqs)[::-1])



def random_mask(text_ids):
    """随机mask
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8:
            input_ids.append(4)
            output_ids.append(i)
        elif r < 0.15 * 0.9:
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.15:
            input_ids.append(np.random.choice(len(tokens)) + 7)
            output_ids.append(i)
        else:
            input_ids.append(i)
            output_ids.append(0)
    return input_ids, output_ids


def random_mask_bi(text_ids):
    """随机mask bi-gram
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    tag =0
    for r, (i, tid) in zip(rands, enumerate(text_ids)):
        if tag:
            tag =0
            continue
        if r < 0.15 * 0.8:
            input_ids.append(4)
            output_ids.append(tid)
            if i != len(text_ids)-1:
                input_ids.append(4)
                output_ids.append(text_ids[i+1])
                tag =1
        elif r < 0.15 * 0.9:
            input_ids.append(tid)
            output_ids.append(tid)
        elif r < 0.15:
            input_ids.append(np.random.choice(len(tokens)) + 7)
            output_ids.append(tid)
        else:
            input_ids.append(tid)
            output_ids.append(0)
    return input_ids, output_ids


def sample_convert(text1, text2, label, random=False):
    """转换为MLM格式
    """
    text1_ids = [tokens.get(t, 1) for t in text1]
    text2_ids = [tokens.get(t, 1) for t in text2]
    if random:
        if np.random.random() < 0.5:
            text1_ids, text2_ids = text2_ids, text1_ids
        if np.random.random() <= 0.8:
            text1_ids, out1_ids = random_mask(text1_ids)
            text2_ids, out2_ids = random_mask(text2_ids)
        else:
            text1_ids, out1_ids = random_mask_bi(text1_ids)
            text2_ids, out2_ids = random_mask_bi(text2_ids)
    else:
        out1_ids = [0] * len(text1_ids)
        out2_ids = [0] * len(text2_ids)
    token_ids = [2] + text1_ids + [3] + text2_ids + [3]
    segment_ids = [0] * len(token_ids)
    output_ids = [label + 5] + out1_ids + [0] + out2_ids + [0]
    return token_ids, segment_ids, output_ids


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids, output_ids = sample_convert(
                text1, text2, label,random #加入random时会让训练数据进行mask
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(output_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids], batch_output_ids
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []

                


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_mlm=True,
    keep_tokens=[0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)],
    model = 'nezha',

)
model.load_weights('./pretrain_model/nezha2/q2q1nezha5e-5.weights')
# print(model.bert.outputs)

 
def masked_crossentropy(y_true, y_pred):
    """mask掉非预测部分
    """
    y_true = K.reshape(y_true, K.shape(y_true)[:2])
    y_mask = K.cast(K.greater(y_true, 0.5), K.floatx())
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = K.sum(loss * y_mask) / K.sum(y_mask)
    return loss[None, None]



AdamW = extend_with_weight_decay(Adam)
AdamWG = extend_with_gradient_accumulation(AdamW)
AdamWGE = extend_with_exponential_moving_average(AdamWG)
# AdamWGEL = extend_with_lookahead(AdamWGE)#加入lookahead
AdamWGELR = extend_with_piecewise_linear_lr(AdamWGE)

opt = AdamWGELR(learning_rate=1e-4, exclude_from_weight_decay=['Norm', 'bias'], grad_accum_steps=2,lr_schedule={1500: 1, 3000: 0.1})  #梯度累计改为2

model.compile(loss=masked_crossentropy, optimizer=opt)


model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)




def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(model, 'Embedding-Token', 0.5)




def evaluate(data):
    """线下评测函数
    """
    Y_true, Y_pred = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)[:, 0,5:7]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        y_true = y_true[:, 0] - 5
        Y_pred.extend(y_pred)
        Y_true.extend(y_true)
    return roc_auc_score(Y_true, Y_pred)




class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_score = evaluate(valid_generator)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            model.save_weights('./pretrain_model/nezha2/finetune/pet_q2q1nezha5e-5.weights')
        print(
            u'val_score: %.5f, best_val_score: %.5f\n' %
            (val_score, self.best_val_score)
        )


def predict_to_file(out_file):
    """预测结果到文件
    """
    F = open(out_file, 'w')
    for x_true, _ in tqdm(test_generator):
        y_pred = model.predict(x_true)[:, 0, 5:7]
#         y_pred = model.predict(x_true)[:,1]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        for p in y_pred:
            F.write('%f\n' % p)
    F.close()




if __name__ == '__main__':

    evaluator = Evaluator()
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
    # predict test and write to file
    model.load_weights('./pretrain_model/nezha2/finetune/pet_q2q1nezha5e-5.weights')
    predict_to_file('0326pet_bigramnezha_adv_10epoch.csv')


# -

