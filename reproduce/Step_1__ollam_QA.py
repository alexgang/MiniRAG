# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import ollama_model_complete, ollama_embed
from minirag.utils import EmbeddingFunc
from transformers import AutoModel,AutoTokenizer
from datetime import datetime
import asyncio
from openai import OpenAI  # 添加OpenAI客户端导入

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse
def get_args(): 
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument('--model', type=str, default='nemotron-mini')
    parser.add_argument('--outputpath', type=str, default='./logs/Default_output-nemotron.csv')
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
    LLM_MODEL = "qwen2.5:1.5b"
elif args.model == 'deepseek':
    LLM_MODEL = "deepseek-r1:1.5b"
elif args.model == 'nemotron-mini' or args.model == 'nemotron-mini-instruct':               
    LLM_MODEL = "nemotron-mini"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name=LLM_MODEL,
    llm_model_max_async=1,#4,
    # llm_model_func=gpt_4o_mini_complete,
    llm_model_max_token_size=8192,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 8192}},
    #llm_model_name= LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: ollama_embed(
            texts, embed_model="all-minilm", host="http://localhost:11434"
        )
    ),
)

#Now QA
QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode='r', encoding='utf-8') as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row['Question'])
        GA_LIST.append(row['Gold Answer'])

# 添加OpenAI客户端初始化
client = OpenAI(
    api_key="sk-s1VhTMnkeP6TtXjNwYc4NFXKBRSBGSeW06roomw1vcBC0DfC",  # 替换为您的API key
    base_url="https://api.chatanywhere.tech/v1"  # 替换为您的base URL
)

async def run_query(rag: MiniRAG, question: str) -> str:
    try:
        print(f"\n尝试查询: {question}")
        # 只使用 mode 参数
        param = QueryParam(mode="mini")
        
        # 直接使用 aquery 方法
        try:
            result = await rag.aquery(
                question,
                param=param
            )
            if result:
                return result.replace("\n", "").replace("\r", "")
            else:
                return "生成答案失败，请重试。"
        except Exception as e:
            print(f"查询过程出错: {str(e)}")
            return f"查询出错: {str(e)}"
            
    except Exception as e:
        print(f'Error in run_query: {str(e)}')
        print(f'Error type: {type(e)}')
        import traceback
        print(f'Traceback: {traceback.format_exc()}')
        return "查询系统出现错误，请稍后重试。"

async def compare_answers(gold_answer: str, generated_answer: str) -> tuple[bool, float, str]:
    """比较标准答案和生成答案的相似度"""
    # 检查答案是否为空或错误信息
    if not generated_answer or "抱歉" in generated_answer or "Error" in generated_answer:
        return False, 0.0, "生成的答案无效或为空"
        
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 改用 gpt-3.5-turbo，因为 gpt-4o-mini 可能不存在
            messages=[
                {"role": "system", "content": "你是一个专业的答案评估专家。你需要判断两个答案是否表达相同的意思，请关注答案的结论是否一致，分析过程的差异可以忽略，并给出相似度评分（0-100）。"},
                {"role": "user", "content": f"""请分析以下两个答案是否表达相同的意思：
                标准答案：{gold_answer}
                生成答案：{generated_answer}
                
                请按以下格式输出：
                相似: [True/False]
                相似度: [0-100]
                分析理由: [简要分析]"""}
            ]
        )
        
        result = response.choices[0].message.content
        print(f"OpenAI 原始响应: {result}")  # 添加调试信息
        
        # 解析输出
        lines = result.split('\n')
        is_similar = 'True' in lines[0].lower()
        try:
            similarity = float(lines[1].split(':')[1].strip())
        except (IndexError, ValueError):
            similarity = 0.0
            print(f"Warning: Could not parse similarity score from response: {result}")
        
        try:
            reason = lines[2].split(':')[1].strip()
        except (IndexError, ValueError):
            reason = "No analysis provided"
            print(f"Warning: Could not parse analysis from response: {result}")
        
        return is_similar, similarity, reason
    except Exception as e:
        print(f'Error in compare_answers: {e}')
        return False, 0.0, f"Error: {str(e)}"

async def run_experiment(output_path):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    headers = ['Question', 'Gold Answer', 'minirag', 'Is Similar', 'Similarity Score', 'Analysis']
    total_questions = 0
    similar_answers = 0

    q_already = []
    if os.path.exists(output_path):
        with open(output_path, mode='r', encoding='utf-8') as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.append(row['Question'])
                if 'Is Similar' in row and row['Is Similar'].lower() == 'true':
                    total_questions += 1
                    similar_answers += 1

    row_count = len(q_already)
    print('row_count', row_count)
    
    with open(output_path, mode='a', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)

        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):
            try:
                QUESTION = QUESTION_LIST[QUESTIONid]
                Gold_Answer = GA_LIST[QUESTIONid]
                print(f"\n处理问题 {QUESTIONid + 1}/{len(QUESTION_LIST)}")
                print('问题:', QUESTION)
                print('标准答案:', Gold_Answer)

                # 添加重试机制
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        minirag_answer = await run_query(rag, QUESTION)
                        if minirag_answer and "错误" not in minirag_answer:
                            break
                        print(f"第 {attempt + 1} 次尝试失败，等待重试...")
                        await asyncio.sleep(2)  # 等待2秒后重试
                    except Exception as e:
                        print(f"第 {attempt + 1} 次尝试出错: {str(e)}")
                        if attempt == max_retries - 1:
                            minirag_answer = "多次尝试后仍然失败"
                        else:
                            await asyncio.sleep(2)
                            continue

                print('生成答案:', minirag_answer)
                
                # 比较答案相似度
                is_similar, similarity, analysis = await compare_answers(Gold_Answer, minirag_answer)
                print(f'相似度: {similarity}%, 是否相似: {is_similar}')
                print(f'分析: {analysis}')
                
                # 更新统计
                total_questions += 1
                if is_similar:
                    similar_answers += 1
                
                # 计算当前准确率
                accuracy = (similar_answers / total_questions) * 100
                print(f'当前准确率: {accuracy:.2f}%')
                
                writer.writerow([
                    QUESTION, 
                    Gold_Answer, 
                    minirag_answer, 
                    is_similar,
                    similarity,
                    analysis
                ])
                
                # 强制写入文件
                log_file.flush()
                
            except Exception as e:
                print(f"处理问题时出错: {str(e)}")
                continue

    final_accuracy = (similar_answers / total_questions) * 100
    print(f'Experiment completed. Final Accuracy: {final_accuracy:.2f}%')
    print(f'Total Questions: {total_questions}')
    print(f'Similar Answers: {similar_answers}')
    print(f'Experiment data has been recorded in the file: {output_path}')

async def main():
    # 可以在这里添加初始化代码
    await run_experiment(OUTPUT_PATH)

if __name__ == "__main__":
    asyncio.run(main())
# # Perform naive search
# print(
#     rag.query("详细说明一下lihua最近这一年的生活怎么样?发生了哪些重要的事情?", param=QueryParam(mode="mini"))
# )