from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from dotenv import load_dotenv

# 导入自定义模块 (Import custom modules)
# 注意：请确保文件名与导入名称一致 (load_model.py, quiz_core.py, 等)
from load_model import search_engine
from quiz_core import generate_quiz_question, verify_quiz_answer
from punctuate_core import punctuate_classic_text
from translate_core import translate_classic_text

# 加载环境变量 (Load environment variables from .env file)
load_dotenv()

# 初始化 FastAPI 应用
app = FastAPI(
    title="古诗词教学与文本挖掘平台接口",
    description="整合了检索、猜注释、自动断句及翻译功能的后端API",
    version="1.0.0"
)

# 配置 CORS (允许跨域请求)
# 允许前端 (HTML/JS) 从任意端口访问后端接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. 检索功能接口 (Search API)
# ==========================================
@app.post("/api/classic/search")
async def api_search(request_data: dict):
    """
    根据用户输入的查询词检索古诗词。
    返回匹配的诗句、作者、朝代及原文内容。
    """
    try:
        query_text = request_data.get("query", "").strip()
        if not query_text:
            return {"success": False, "message": "请输入检索关键词"}

        # 调用搜索引擎 (load_model.py)
        # 默认返回 Top-5 结果
        json_result = search_engine(query_text, top_k=5)
        full_results = json.loads(json_result)
        
        # 筛选前端需要的字段
        filtered_results = []
        for item in full_results:
            filtered_item = {
                "matched_sentence": item.get("matched_sentence", ""),
                "name": item.get("name", ""),
                "author": item.get("author", ""),
                "dynasty": item.get("dynasty", ""),
                "content": item.get("content", "")
            }
            filtered_results.append(filtered_item)

        return {"success": True, "list": filtered_results}
    except Exception as e:
        return {"success": False, "message": f"服务器内部错误 (Server Error): {str(e)}"}

# ==========================================
# 2. 猜注释游戏接口 (Quiz API)
# ==========================================
@app.get("/api/classic/quiz/get")
async def api_get_quiz():
    """
    生成一个新的猜注释题目。
    从 output.xlsx 文件中读取数据。
    """
    try:
        # 生成题目 (quiz_core.py)
        # 确保 output.xlsx 文件在 backend 目录下
        question = generate_quiz_question(excel_path="output.xlsx")
        
        if "error" in question:
            return {"success": False, "message": question["error"]}
            
        return {
            "success": True,
            "text": question["text"],
            "targetWord": question["targetWord"],
            "correctAnswer": question["correctAnswer"],
            "title": question["title"],
            "author": question["author"]
        }
    except Exception as e:
        return {"success": False, "message": f"服务器内部错误 (Server Error): {str(e)}"}

@app.post("/api/classic/quiz/submit")
async def api_submit_quiz(request_data: dict):
    """
    提交用户的答案，并进行自动评分和反馈。
    """
    try:
        target_word = request_data.get("targetWord", "").strip()
        user_answer = request_data.get("userAnswer", "").strip()
        correct_answer = request_data.get("correctAnswer", "").strip()

        if not target_word or not user_answer or not correct_answer:
            return {"success": False, "message": "缺少必要参数 (Missing required parameters)"}

        # 验证答案 (quiz_core.py)
        result = verify_quiz_answer(user_answer, correct_answer, target_word)
        
        if not result["success"]:
             return {"success": False, "message": result["feedback"]}

        # 返回详细的评分结果
        return {
            "success": True,
            "score": result["score"],
            "isCorrect": result["isCorrect"],
            "correctAnswer": result["correctAnswer"],
            "entity_score": result["entity_score"],
            "penalty": result["penalty"],
            "error_count": result["error_count"],
            "user_entities": result["user_entities"],
            "correct_entities": result["correct_entities"],
            "user_relations": result["user_relations"],
            "correct_relations": result["correct_relations"],
            "error_relations": result["error_relations"],
            "matched_entity_pairs": result["matched_entity_pairs"],
            "feedback": result["feedback"]
        }
    except Exception as e:
        return {"success": False, "message": f"服务器内部错误 (Server Error): {str(e)}"}

# ==========================================
# 3. 自动断句接口 (Automatic Punctuation API)
# ==========================================
@app.post("/api/classic/punctuate")
async def api_classic_punctuate(request_data: dict):
    """
    调用大模型对输入的古文进行自动断句。
    """
    try:
        raw_text = request_data.get("text", "").strip()
        if not raw_text:
            return {"success": False, "message": "请输入需要断句的文本"}

        # 调用断句核心功能 (punctuate_core.py)
        punctuated_result = punctuate_classic_text(raw_text)
        return {"success": True, "punctuatedText": punctuated_result}
    except Exception as e:
        return {"success": False, "message": f"服务器内部错误 (Server Error): {str(e)}"}

# ==========================================
# 4. 自动翻译接口 (Translation API)
# ==========================================
@app.post("/api/classic/translate")
async def api_translate(request_data: dict):
    """
    调用大模型将古文翻译为现代汉语。
    """
    try:
        raw_text = request_data.get("text", "").strip()
        if not raw_text:
            return {"success": False, "message": "请输入需要翻译的文本"}

        # 调用翻译核心功能 (translate_core.py)
        translated_result = translate_classic_text(raw_text)
        return {"success": True, "translation": translated_result}
    except Exception as e:
        return {"success": False, "message": f"服务器内部错误 (Server Error): {str(e)}"}

# 启动服务
if __name__ == "__main__":
    print("正在启动古诗词教学平台后端服务...")
    print("服务端口: 8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
