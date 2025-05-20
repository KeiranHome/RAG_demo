import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("QIANWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

def build_ivf_index(embeddings: np.ndarray, nlist: int = 100) -> faiss.Index:
    """
    构建IVF索引，适合百万级以上向量检索
    nlist 是倒排表的数量，调节索引的粗细程度
    """
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(embeddings)
    index.add(embeddings)
    return index

def embed_text(text: str, model="text-embedding-v3") -> List[float]:
    """
    使用通义千问API生成文本的embedding向量
    """
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def search_index(index: faiss.Index, query_vec: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    在FAISS索引中进行向量搜索
    """
    if isinstance(index, faiss.IndexIVF) and not index.is_trained:
        raise ValueError("IVF索引尚未训练，无法进行搜索。请先训练后再检索。")
    distances, indices = index.search(query_vec, top_k)
    return distances[0], indices[0]

def load_data_group(folder_path: str, use_saved_index: bool = False, nlist: int = 100) -> Tuple[faiss.Index, List[Dict]]:
    """
    加载一个数据组，包含 .index, _chunks.json, _vectors.json 三个文件
    并且将文本与向量对应起来，返回FAISS索引和文本元数据列表。
    如果 use_saved_index 为 True 且存在 merged_ivf.index 文件，则直接加载。
    """
    saved_index_path = os.path.join(folder_path, "merged_ivf.index")
    metadata_path = os.path.join(folder_path, "merged_metadata.json")

    if use_saved_index and os.path.exists(saved_index_path) and os.path.exists(metadata_path):
        print("加载已保存的FAISS IVF索引...")
        index = faiss.read_index(saved_index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    # 否则重新加载向量数据并构建索引
    files = os.listdir(folder_path)
    index_files = [f for f in files if f.endswith('.index')]

    all_embeddings = []
    all_metadata = []

    for idx_file in index_files:
        prefix = idx_file[:-6]
        chunks_file = prefix + "_chunks.json"
        vectors_file = prefix + "_vectors.json"

        idx_path = os.path.join(folder_path, idx_file)
        chunks_path = os.path.join(folder_path, chunks_file)
        vectors_path = os.path.join(folder_path, vectors_file)

        if not (os.path.exists(chunks_path) and os.path.exists(vectors_path)):
            print(f"警告：缺少 {chunks_file} 或 {vectors_file}，跳过 {prefix}")
            continue

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        with open(vectors_path, "r", encoding="utf-8") as f:
            vectors_data = json.load(f)

        id_to_text = {item['id']: item['text'] for item in chunks_data}
        for vec_item in vectors_data:
            vec_id = vec_item['id']
            embedding = vec_item.get('embedding')
            if embedding is None:
                print(f"警告：id={vec_id} 缺少embedding，跳过")
                continue
            text = id_to_text.get(vec_id, "")
            all_embeddings.append(embedding)
            all_metadata.append({
                "id": vec_id,
                "text": text,
                "source_file": vec_item.get("source_file", ""),
                "prefix": prefix
            })

    if len(all_embeddings) == 0:
        raise ValueError("没有加载到任何embedding向量，无法构建索引")

    dimension = len(all_embeddings[0])
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    print(f"总共加载向量数：{len(all_embeddings)}，向量维度：{dimension}")

    index = build_ivf_index(embeddings_np, nlist=nlist)
    print(f"IVF索引构建完成，向量总数: {index.ntotal}")

    # 保存索引与元数据
    faiss.write_index(index, saved_index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    return index, all_metadata

class RAGEngine:
    def __init__(self, data_folder: str, embedding_model: str = "text-embedding-v3", nlist: int = 100, nprobe: int = 10):
        self.embedding_model = embedding_model
        self.index, self.metadata = load_data_group(data_folder, use_saved_index=True, nlist=nlist)
        self.chat_history = []
        if isinstance(self.index, faiss.IndexIVF):
            self.index.nprobe = nprobe  # 提高召回率

    def post_process_answer(self, answer: str) -> str:
        answer = answer.strip()
        max_length = 2000
        if len(answer) > max_length:
            answer = answer[:max_length] + "……【回答被截断】"
        return answer

    def query(self, question: str, top_k: int = 5, multi_turn: bool = False) -> str:
        query_emb = embed_text(question, model=self.embedding_model)
        query_vec = np.array([query_emb], dtype=np.float32)
        distances, indices = search_index(self.index, query_vec, top_k)

        context_texts = []
        for idx in indices:
            if 0 <= idx < len(self.metadata):
                context_texts.append(self.metadata[idx]['text'])
        context = "\n\n".join(context_texts)

        if multi_turn and self.chat_history:
            messages = self.chat_history.copy()
            messages.append({"role": "user", "content": f"根据以下内容，回答问题：\n\n{context}\n\n问题：{question}"})
        else:
            messages = [{"role": "user", "content": f"请根据以下资料内容，结合您的知识，详细回答下面的问题：\n\n{context}\n\n问题：{question}"}]

        response = client.chat.completions.create(
            model="qwen1.5-14b-chat",
            messages=messages
        )
        answer = response.choices[0].message.content.strip()
        answer = self.post_process_answer(answer)

        if multi_turn:
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})

        return answer

if __name__ == "__main__":
    folder = "../Datas/Output"  # 修改为你的文件夹路径
    rag_engine = RAGEngine(folder)

    print("欢迎使用RAG问答系统，输入exit退出")
    while True:
        q = input("\n请输入你的问题：")
        if q.strip().lower() in ("exit", "quit", "退出"):
            print("感谢使用，程序退出。")
            break
        if not q.strip():
            print("问题不能为空，请重新输入。")
            continue
        ans = rag_engine.query(q, top_k=5)
        print("\n=== 回答 ===\n")
        print(ans)
