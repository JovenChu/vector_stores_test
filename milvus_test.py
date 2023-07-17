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
@Desc    :   基于hello_milvus.py修改，用于测试服务器的向量数据库，embedding模型使用text2vec-base，可以换其他的
'''

import time
import numpy as np
import pandas as pd
import random
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import torch.cuda
import torch.backends
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
embedding_model_dict = {
    "text2vec-base": "models/text2vec-base-chinese",
}
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dim =768

# 加载向量模型
embedding_model = "text2vec-base"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': EMBEDDING_DEVICE})

# #################################################################################
# 0、生成文本数据
# 加载向量模型
embedding_model = "text2vec-base"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': EMBEDDING_DEVICE})
result = []
vectors = []

# 生成向量
start_time = time.time()
with open('datas/train.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        items = line.split('\t')
        if len(items)==3:
            if items[0]:
                result.append(items[0])
                vectors.append(embeddings.embed_query(items[0]))
            if items[1]:
                result.append(items[1])
                vectors.append(embeddings.embed_query(items[1]))
end_time = time.time()
print("生成80万条文本向量耗时：{} 秒".format(str(end_time-start_time)))

counts = [i for i in range(len(result))]
print(len(counts))
print("保存成csv")
# 保存成csv
alls = {
    "ids": counts,
    "sentences": result,
    "vectors": vectors
}
from pandas.core.frame import DataFrame
all_df = DataFrame(alls)
all_df.to_csv("datas/train.csv", index=False)

#################################################################################
# 1. connect to Milvus——连接向量数据库
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("text2vec_test80")
print(f"Does collection text2vec_test80 exist in Milvus: {has}")

# #################################################################################
# 2. create collection——创建表
fields = [
    FieldSchema(name="ids", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "text2vec_test80 is the simplest demo to introduce the APIs")

print(fmt.format("Create collection `text2vec_test80`"))
hello_milvus = Collection("text2vec_test80", schema, consistency_level="Strong")

# ################################################################################
# 3. insert data——插入数据
# We are going to insert 3000 rows of data into `hello_milvus`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

# 读取数据
print(fmt.format("Start inserting entities"))
csv_df = pd.read_csv("datas/train.csv")  # 读取训练数据
# csv_data = csv_df.loc[:150000]
# csv_data = csv_df.loc[150001:300000]
# csv_data = csv_df.loc[300001:450000]
csv_data = csv_df.loc[450001:]
print(csv_data.shape)  # (476066, 3)

num_entities = len(csv_data)
# print(num_entities)
# print(csv_data["sentences"][0])
vectors = list(csv_data["vectors"])
now_vectors = []
for i in vectors:
    # 把列表样式的字符串重新转换为列表
    now_vectors.append(eval(i))
print(type(now_vectors[0]))
# print(now_vectors[0])

entities = [
    # provide the pk field because `auto_id` is set to False
    [int(id) for id in csv_data["ids"]],
    np.array(now_vectors)
]
# 数据插入
start_time = time.time()
insert_result = hello_milvus.insert(entities)
end_time = time.time()
print("插入50万条文本向量耗时：{} 秒".format(str(end_time-start_time)))
# 插入50万条文本向量耗时：36.61300301551819 秒 + 33.45749497413635 秒 + 32.70834016799927 秒 = 100 秒

print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites

# ################################################################################
# 4. create index——创建索引
# We are going to create an IVF_FLAT index for hello_milvus collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 4096},
}
# 构建索引
start_time = time.time()
hello_milvus.create_index("embeddings", index)
end_time = time.time()
print("构建50万向量的索引耗时：{} 秒".format(str(end_time-start_time)))
# 构建50万向量的索引耗时：52.08765196800232 秒

################################################################################
# 5. search, query, and hybrid search——检索测试
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
print(fmt.format("Start loading"))
# 加载数据
start_time = time.time()
hello_milvus = Collection("text2vec_test80")
hello_milvus.load()
end_time = time.time()
print("加载数据耗时：{} 秒".format(str(end_time-start_time)))
# 加载数据耗时：0.008812189102172852 秒
# 加载50万数据耗时：22.79383635520935 秒
# 参数
# search_params = {
#     "metric_type": "l2",
#     "params": {"nprobe": 32},
# }
search_params = {"metric_type": "L2", "params": {"nprobe": 32}}

# 查询
print(fmt.format("Start searching based on vector similarity"))
while True:
    search_sen = input("请输入查询语句：")
    search_vec = np.array([embeddings.embed_query(search_sen)])

    start_time = time.time()
    result = hello_milvus.search(search_vec, "embeddings", search_params, limit=3, output_fields=["ids"])
    end_time = time.time()

    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, ids field: {hit.entity.get('ids')}")
    print(search_latency_fmt.format(end_time - start_time))


""" 1万数据平均查询耗时200ms，样例如下：
=== Start searching based on vector similarity ===

请输入查询语句：一架飞机正在起飞。
hit: (distance: 0.0, id: 1), ids field: 1
hit: (distance: 30.937931060791016, id: 0), ids field: 0
hit: (distance: 76.99212646484375, id: 2620), ids field: 2620
search latency = 0.2920s
请输入查询语句：一个人把一只猫扔在天花板上。
hit: (distance: 0.0, id: 19), ids field: 19
hit: (distance: 11.008106231689453, id: 18), ids field: 18
hit: (distance: 68.62686920166016, id: 1363), ids field: 1363
search latency = 0.2326s
"""


"""50万数据平均查询耗时200ms
加载50万数据耗时：22.79383635520935 秒

=== Start searching based on vector similarity ===

请输入查询语句：一架飞机正在起飞。
hit: (distance: 49.9296989440918, id: 313925), ids field: 313925
hit: (distance: 50.9527587890625, id: 233208), ids field: 233208
hit: (distance: 65.26101684570312, id: 14035), ids field: 14035
search latency = 0.2701s
请输入查询语句：一个人把一只猫扔在天花板上。
hit: (distance: 184.68429565429688, id: 232041), ids field: 232041
hit: (distance: 185.34658813476562, id: 194253), ids field: 194253
hit: (distance: 199.0066680908203, id: 299685), ids field: 299685
search latency = 0.1941s

"""