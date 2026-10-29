import torch
import pickle
import numpy as np
import os
import re
import json
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# [配置 - Configuration]
# 自动获取当前文件所在目录，实现跨平台相对路径 (Auto-detect relative path)
# ==============================================================================

# 获取当前文件 (load_model.py) 的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. BERT 模型路径 (相对于当前文件的路径)
# 假设 bert-ancient-chinese 文件夹就在 backend 目录下
BERT_MODEL_PATH = os.path.join(BASE_DIR, "bert-ancient-chinese")

# 2. Embedding (.pkl) 文件夹路径
# 假设 embeddings_output 文件夹就在 backend 目录下
EMBEDDING_DIR_PATH = os.path.join(BASE_DIR, "embeddings_output")

# 3. 原始数据 (.json) 文件路径
# 假设 json 文件就在 backend 目录下
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "古诗词数据.json")

# ==============================================================================
# [缓存 - Cache]
# ==============================================================================
CACHE = {
    "model": None,  # (tokenizer, model, device)
    "warehouse": {},  # { window_size: (matrix, metadata) }
    "original_data": None  # { id: { title, author, ... } }
}


# ==============================================================================
# [核心函数 - Core Functions]
# ==============================================================================

def load_original_data():
    """
    加载原始 JSON 数据并构建 ID 映射表，以便快速查询详情
    """
    if CACHE["original_data"] is not None:
        return CACHE["original_data"]

    print("[INFO] Loading original JSON data...")
    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"[ERROR] Original data file not found: {ORIGINAL_DATA_PATH}")
        return {}

    try:
        with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)

        # List -> Dict Mapping (Key: id)
        data_map = {}
        for item in raw_list:
            # 确保 id 存在且为 int/str 统一格式
            pid = item.get('id')
            if pid is not None:
                data_map[pid] = item

        CACHE["original_data"] = data_map
        print(f"[INFO] Original data loaded. Total records: {len(data_map)}")
        return data_map
    except Exception as e:
        print(f"[ERROR] Failed to load JSON data: {e}")
        return {}


def get_model():
    """
    加载 BERT 模型 (单例模式)
    """
    if CACHE["model"] is not None:
        return CACHE["model"]

    print("[INFO] Initializing BERT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, local_files_only=True)
        model = BertModel.from_pretrained(BERT_MODEL_PATH, local_files_only=True)
        model.to(device)
        model.eval()

        CACHE["model"] = (tokenizer, model, device)
        print("[INFO] BERT model loaded successfully.")
        return CACHE["model"]
    except Exception as e:
        print(f"[ERROR] Failed to load BERT model: {e}")
        return None


def load_warehouse(window_size):
    """
    加载指定窗口大小的 pkl 数据
    """
    if window_size in CACHE["warehouse"]:
        return CACHE["warehouse"][window_size]

    pkl_path = os.path.join(EMBEDDING_DIR_PATH, f"embeddings_window_{window_size}.pkl")
    print(f"[INFO] Loading embedding warehouse: Window_{window_size} ...")

    if not os.path.exists(pkl_path):
        print(f"[ERROR] Warehouse file not found: {pkl_path}")
        return None, None

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        vectors = []
        metadata = []
        for pid, items in data.items():
            for item in items:
                vectors.append(item['vector'])
                metadata.append({'id': pid, 'text': item['text']})

        matrix = np.array(vectors)
        CACHE["warehouse"][window_size] = (matrix, metadata)
        print(f"[INFO] Warehouse loaded. Size: {len(matrix)}")
        return matrix, metadata
    except Exception as e:
        print(f"[ERROR] Failed to load warehouse: {e}")
        return None, None


def determine_window_size(text):
    """
    根据标点数量决定窗口大小
    规则: 0 -> Win1, 1 -> Win2, 2 -> Win3, >=3 -> Win4
    """
    punctuations = set("，。？！")
    count = sum(1 for char in text if char in punctuations)

    if count == 0:
        target = 1
    elif count == 1:
        target = 2
    elif count == 2:
        target = 3
    else:
        target = 4

    return target, count


def clean_text(text):
    """去除末尾标点"""
    return re.sub(r'[，。？！]+$', '', text)


def search_engine(query_text, top_k=5):
    """
    主搜索入口
    """
    # 1. 准备模型
    model_pack = get_model()
    if not model_pack: return json.dumps({"error": "Model load failed"}, ensure_ascii=False)
    tokenizer, model, device = model_pack

    # 2. 准备原始数据 (详情)
    original_data = load_original_data()

    # 3. 决定窗口
    target_window, punc_count = determine_window_size(query_text)
    print(f"[INFO] Query analysis: {punc_count} punctuations -> Use Window_{target_window}")

    # 4. 加载向量仓库
    matrix, meta = load_warehouse(target_window)
    if matrix is None: return json.dumps({"error": "Warehouse load failed"}, ensure_ascii=False)

    # 5. 生成查询向量
    inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean Pooling
    token_embeddings = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    query_vec = (sum_embeddings / sum_mask)[0].cpu().numpy().reshape(1, -1)

    # 6. 计算相似度 (多取一些以便去重)
    # 取 top_k * 5 个候选，确保去重后能剩下足够的数量
    CANDIDATE_POOL_SIZE = top_k * 5
    scores = cosine_similarity(query_vec, matrix)[0]
    candidate_indices = scores.argsort()[::-1][:CANDIDATE_POOL_SIZE]

    # 7. 结果过滤与组装 (去重 + 详情合并)
    final_results = []
    seen_ids = set()

    for idx in candidate_indices:
        if len(final_results) >= top_k:
            break

        pid = meta[idx]['id']

        # [去重逻辑] 同一首诗只取最高分的一个片段
        if pid in seen_ids:
            continue

        seen_ids.add(pid)

        # 获取原始详情
        detail = original_data.get(pid, {})

        result_item = {
            "similarity": float(f"{scores[idx]:.4f}"),  # 相似度
            "poem_id": pid,  # 诗词ID
            "matched_sentence": clean_text(meta[idx]['text']),  # 匹配到的句子 (Cleaned)

            # 原始数据字段
            "name": detail.get("name", ""),
            "author": detail.get("author", ""),
            "dynasty": detail.get("dynasty", ""),
            "category": detail.get("category", ""),
            "content": detail.get("content", ""),
            "new_poem_theme": detail.get("new_poem_theme", "")
        }
        final_results.append(result_item)

    # 返回 JSON 格式字符串 (indent=4 for pretty print)
    return json.dumps(final_results, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    user_query = "落日"
    # 执行搜索
    json_result = search_engine(user_query, top_k=3) # 调整你想看的参数
    # 打印最终 JSON
    print(json_result)