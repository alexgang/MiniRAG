# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import csv
import asyncio
import numpy as np
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import ollama_model_complete, ollama_embed,ollama_model_if_cache
from minirag.utils import EmbeddingFunc
from transformers import AutoModel,AutoTokenizer
from datetime import datetime

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
import argparse

def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument('--model', type=str, default='nemotron-mini')
    parser.add_argument('--outputpath', type=str, default='./logs/Default_output.csv')
    parser.add_argument('--workingdir', type=str, default='./LiHua-World')
    parser.add_argument('--datapath', type=str, default='./dataset/LiHua-World/data/')
    parser.add_argument('--querypath', type=str, default='./dataset/LiHua-World/qa/query_set.csv')
    args = parser.parse_args()
    return args

args = get_args()


if args.model == 'PHI':
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
elif args.model == 'GLM':
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == 'MiniCPM':
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == 'qwen':
    LLM_MODEL = "qwen2.5:3bm"
elif args.model == 'deepseek':
    LLM_MODEL = "deepseek-r1:1.5b"
elif args.model == 'nemotron-mini': 
    LLM_MODEL = "nemotron-mini"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
#WORKING_DIR = "./little-prince"
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await ollama_model_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await ollama_embed(
        texts,
        embed_model="all-minilm",
        host="http://localhost:11434",
    )

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    llm_model_name=LLM_MODEL,
    llm_model_max_async=1,
    llm_model_max_token_size=200,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 8192}},
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=embedding_func
    ),
)

async def test_funcs():
    result = await llm_model_func("你好，介绍一下你自己。")
    print("llm_model_func: ", result)

    result = await embedding_func(["你好，介绍一下你自己。"])
    print("embedding_func: ", result)


async def process_files():
    WEEK_LIST = find_txt_files(DATA_PATH)
    for WEEK in WEEK_LIST:
        id = WEEK_LIST.index(WEEK)
        print(f"{id}/{len(WEEK_LIST)}")
        with open(WEEK, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

async def main():
    await test_funcs()
    await process_files()

if __name__ == "__main__":
    asyncio.run(main())

# Perform naive search
#  print(
#    rag.query("小王子和国王是什么关系，他们之间发生了什么事情啊？", param=QueryParam(mode="mini"))
#)  