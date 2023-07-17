#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │ 4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|  │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│ ' │ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘
@File    :   faiss_test.py
@Time    :   2023/07/14 11:11:01
@Author  :   Joven Chu 
@github  :   https://github.com/JovenChu
@Desc    :   测试faiss数据库
'''

import faiss
import pandas as pd
import numpy as np
import time
import torch.cuda
import torch.backends
import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embedding_model_dict = {
    "text2vec-base": "models/text2vec-base-chinese",
}
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dim =768

# 加载向量模型
embedding_model = "text2vec-base"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],model_kwargs={'device': EMBEDDING_DEVICE})

search_latency_fmt = "search latency = {:.4f}s"

# 读取数据
print("开始读取数据......")
# csv_df = pd.read_csv("datas/train.csv")  # 读取训练数据
# csv_data = csv_df.loc[:15]
csv_data = pd.read_csv("datas/train.csv")  # 读取训练数据
print(csv_data.shape)  # (476066, 3)

num_entities = len(csv_data)
print(num_entities)
print(csv_data["sentences"][0])
vectors = list(csv_data["vectors"])
now_vectors = []
for i in vectors:
    # 把列表样式的字符串重新转换为列表
    now_vectors.append(eval(i))
print(type(now_vectors[0]))
print(now_vectors[0])
now_vectors = np.array(now_vectors)

print("开始插入向量数据......")
dimension = now_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
start_time = time.time()
index.add(now_vectors)
end_time = time.time()
print(search_latency_fmt.format(end_time - start_time))
print("查看索引数据：")
print(index.ntotal)

# 保存
dump_file = 'vectors_file/faiss_all.index'
faiss.write_index(index, dump_file)


print("开始查询")
while True:
    search_sen = input("请输入查询语句：")
    search_vec = np.array([embeddings.embed_query(search_sen)])
    topK = 3
    start_time = time.time()
    D, I = index.search(search_vec, topK)
    end_time = time.time()
    result = csv_data["sentences"].iloc[I[0]]
    print(result)
    print(search_latency_fmt.format(end_time - start_time))

# 单独加载
# index = faiss.read_index('vectors_file/faiss.index')
# search_vec = np.array(search_vec)
# topK = 3
# start_time = time.time()
# D, I = index.search(search_vec, topK)
# end_time = time.time()
# print(D)
# print(I)
# print(search_latency_fmt.format(end_time - start_time))

'''向量规模：476066；向量插入速度：1s，平均查询速度：130ms
请输入查询语句：
0         是的，我想一个洞穴也会有这样的问题
1          我认为洞穴可能会有更严重的问题。
110556       在这些洞穴里事情可能会变糟。
Name: sentences, dtype: object
search latency = 0.1315s
'''