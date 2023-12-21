import torch
import sentencepiece as spm
import sys
import numpy as np


def train_model(fname, prefix):
    spm.SentencePieceTrainer.train(input=fname, model_prefix=prefix, vocab_size=8000)


def load_tokenizer(model_file):
    sp = spm.SentencePieceProcessor()       # 实例化一个Processer对象
    if not sp.load(model_file=model_file):    # 用实例化的对象加载模型
        return False, None
    else:
        return True, sp


def load_file_into_splits(text_file, split_ratio):
    with open(text_file, 'r', encoding='gb18030') as file:
        data = file.read()
    split_idx = int(len(data)*split_ratio)
    return data[:split_idx], data[split_idx:]


def encode_and_save(sp, content, prefix):
    token_ids = sp.encode(content, out_type=int)
    print(f"data split of {prefix} has {len(token_ids)} tokens")
    token_ids = np.array(token_ids, dtype=np.int32)
    token_ids.tofile("{}.dat".format(prefix))


def gen_dataset(text_file, model_file):
    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"Load tokenizer model from:{model_file} failed")
        sys.exit()
    split_ratio = 0.9
    train_text, test_text = load_file_into_splits(text_file, split_ratio)
    encode_and_save(sp, train_text, "train")
    encode_and_save(sp, test_text, "test")


def get_batch(data, batch_size=4):
    win_len = 10
    ix = torch.randint(len(data)-win_len-1, (batch_size,))
    print(ix)
    x = np.stack([data[i:i+win_len] for i in ix])
    y = np.stack([data[i+1:i+1+win_len] for i in ix])
    return x, y


def gen_samples(fname, prefix):
    model_file = prefix + '.model'
    train_data = np.memmap(fname, dtype=np.int32, mode='r')
    x, y = get_batch(train_data)
    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"load tokenizer model from: {model_file} failed")
        sys.exit(1)
    for features, targets in zip(x, y):
        print("features: ", sp.decode(features.tolist()))
        print("targets: ", sp.decode(features.tolist()))


corpus = 'BirdCouple.txt'
prefix = 'BirdCouple'
train_model(corpus, prefix)
gen_dataset(corpus, prefix+".model")

gen_samples("train.dat", prefix)