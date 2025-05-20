# 导入所需模块
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF
import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import markdown  # 新增：用于解析MD文件
from bs4 import BeautifulSoup  # 新增：用于从HTML中提取文本

# 加载环境变量
load_dotenv()

# 初始化通义千问客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 使用通义千问API密钥
    base_url=os.getenv("QIANWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

def split_text_into_chunks(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """将长文本按句子切分为多个chunk，每个不超过max_tokens，支持滑动窗口式重叠"""
    cleaned = re.sub(r'\s+', ' ', text)  # 修正正则表达式，将s+改为\s+
    sentences = re.split(r'(?<=[。！？.!?])', cleaned)

    chunks = []
    chunk = ''
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_tokens:
            chunk += sentence
        else:
            chunks.append(chunk)
            chunk = chunk[-overlap:] + sentence

    if chunk:
        chunks.append(chunk)
    return chunks

def extract_text_from_md(file_path: str) -> str:
    """提取MD文件中的文本内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 将MD转换为HTML
    html = markdown.markdown(md_content)
    
    # 从HTML中提取纯文本
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def save_chunks_to_json(chunks: List[str], output_path: str) -> None:
    """将文本块保存为JSON文件"""
    data = [{"id": i, "text": chunk, "length": len(chunk)} for i, chunk in enumerate(chunks)]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_embeddings(texts: List[str], model: str = "text-embedding-v3", batch_size: int = 10) -> List[List[float]]:
    """使用通义千问API生成文本嵌入向量，支持分批处理"""
    all_embeddings = []
    
    # 分批处理文本
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"正在处理第 {i//batch_size+1}/{(len(texts)+batch_size-1)//batch_size} 批，包含 {len(batch)} 个文本块")
        
        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # 简单的进度反馈
            print(f"第 {i//batch_size+1} 批处理完成")
            
        except Exception as e:
            print(f"处理第 {i//batch_size+1} 批时出错: {e}")
            # 可以选择在此处添加重试逻辑
            return []
    
    return all_embeddings

def save_embeddings_to_json(chunks: List[Dict[str, Any]], embeddings: List[List[float]], output_path: str) -> None:
    """将文本块及其向量保存为JSON文件"""
    if len(chunks) != len(embeddings):
        raise ValueError("文本块数量与向量数量不匹配")
    
    # 将向量添加到元数据中
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def create_faiss_index(embeddings: List[List[float]], index_path: str) -> None:
    """创建并保存FAISS索引"""
    import faiss
    
    # 将Python列表转换为NumPy数组
    vectors = np.array(embeddings, dtype='float32')
    
    # 创建FAISS索引（使用FlatL2进行精确搜索）
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    
    # 添加向量到索引
    index.add(vectors)
    
    # 保存索引
    faiss.write_index(index, index_path)
    print(f"FAISS索引已保存到: {index_path}")

def process_single_file(file_path: str, output_dir: str) -> None:
    """处理单个文件"""
    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取文本
    if file_ext == '.md':
        text = extract_text_from_md(file_path)
    else:
        print(f"不支持的文件类型: {file_ext}")
        return
    
    # 分割文本
    chunks = split_text_into_chunks(text, max_tokens=400, overlap=50)
    
    # 保存文本块
    output_json_path = os.path.join(output_dir, f"{base_name}_chunks.json")
    save_chunks_to_json(chunks, output_json_path)
    print(f"文件 {file_name} 共分成 {len(chunks)} 段")
    print(f"文本块已保存到 {output_json_path}")
    
    # 生成向量
    print(f"正在使用通义千问模型为 {file_name} 生成向量...")
    embeddings = generate_embeddings(chunks)
    
    if embeddings:
        # 转换为字典列表，添加id和length信息
        chunks_with_metadata = [{"id": i, "text": chunk, "length": len(chunk), "source_file": file_name} for i, chunk in enumerate(chunks)]
        
        # 保存带向量的JSON文件
        vector_json_path = os.path.join(output_dir, f"{base_name}_vectors.json")
        save_embeddings_to_json(chunks_with_metadata, embeddings, vector_json_path)
        print(f"向量已保存到 {vector_json_path}")
        
        # 创建并保存FAISS索引
        faiss_index_path = os.path.join(output_dir, f"{base_name}.index")
        create_faiss_index(embeddings, faiss_index_path)
    else:
        print(f"未能为 {file_name} 生成向量")

def batch_process_files(input_dir: str, output_dir: str, file_extensions: List[str] = ['.pdf', '.md']) -> None:
    """批量处理目录中的文件"""
    # 获取所有符合条件的文件
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        print(f"在 {input_dir} 中没有找到符合条件的文件")
        return
    
    print(f"找到 {len(files_to_process)} 个文件需要处理")
    
    # 逐个处理文件
    for i, file_path in enumerate(files_to_process):
        print(f"\n===== 正在处理文件 {i+1}/{len(files_to_process)}: {os.path.basename(file_path)} =====")
        process_single_file(file_path, output_dir)

def main():
    # 配置参数
    input_dir = "../Datas/output_files_v4"  # 输入目录，包含PDF和MD文件
    output_dir = "../Datas/Output"  # 输出目录
    
    # 批量处理文件
    batch_process_files(input_dir, output_dir)


if __name__ == "__main__":
    main()
