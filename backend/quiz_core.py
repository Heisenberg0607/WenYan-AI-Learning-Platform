# quiz_core.py - 基于官方game_api.py重构（匹配README的评分机制）
from game_api import evaluate_answer, get_random_question, extract_entities
import os  

def generate_quiz_question(excel_path="output.xlsx"):
    """
    生成题目数据
    """
    # [추가] 현재 파일(quiz_core.py)이 있는 폴더 경로를 알아냅니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # [추가] 그 경로와 파일명을 합쳐서 완벽한 경로를 만듭니다.
    real_path = os.path.join(current_dir, excel_path)

    try:
        # [수정] 이제 real_path를 넘겨줍니다.
        question = get_random_question(real_path) 
        
        if not question:
            return {"error": "无可用题目 (No questions available)"}

        # 适配前端格式：补充高亮后的诗句（标注目标词条）
        content = question["content"]
        term = question["term"]
        highlighted_text = content.replace(term, f"<span class='target-word'>{term}</span>")

        return {
            "text": highlighted_text,  # 高亮后的诗句（前端显示）
            "targetWord": term,  # 目标词条
            "correctAnswer": question["correct_answer"],  # 标准答案（前端暂存，提交时传回）
            "title": question["poem_name"],  # 诗词名称
            "author": question["author"],  # 作者
            "content": content  # 原始诗词内容
        }
    except Exception as e:
        return {"error": f"生成题目失败：{str(e)}"}


def verify_quiz_answer(user_answer, correct_answer, target_word):
    """
    调用官方evaluate_answer函数验证答案
    :param user_answer: 用户输入的答案
    :param correct_answer: 标准答案（前端传回）
    :param target_word: 目标词条
    :return: 标准化评分结果（适配前端）
    """
    try:
        # 调用官方核心函数（启用BERT，阈值用默认0.7）
        result = evaluate_answer(
            user_answer=user_answer,
            correct_answer=correct_answer,
            head_entity=target_word,
            semantic_threshold=0.7,
            use_bert=True
        )

        # 适配前端的返回格式
        return {
            "score": result["score"],  # 总得分
            "isCorrect": result["is_correct"],  # 是否正确
            "correctAnswer": correct_answer,  # 正确答案
            "entity_score": result["entity_score"],  # 实体匹配得分
            "penalty": result["penalty"],  # 关系错误扣分
            "error_count": result["error_count"],  # 错误关系数量
            "user_entities": result["error_count"],  # 用户答案提取的实体列表
            "correct_entities": result["correct_entities"],  # 标准答案提取的实体列表
            "user_relations": result["user_relations"],  # 用户答案提取的关系列表
            "correct_relations": result["correct_relations"],  # 标准答案提取的关系列表
            "error_relations": result["error_relations"],  # 错误的关系列表
            "matched_entity_pairs": result["matched_entity_pairs"],  # 语义匹配的实体对列表[(user_ent, correct_ent, similarity), ...]
            "feedback": result["feedback"],  # 反馈
            "success": True
        }
    except Exception as e:
        return {
            "score": 0,
            "isCorrect": False,
            "correctAnswer": "",
            "feedback": f"验证答案失败：{str(e)}",
            "success": False
        }


# 测试代码（验证官方函数调用）
if __name__ == "__main__":
    # 测试生成题目
    question = generate_quiz_question()
    print("生成题目：", question)

    # 测试验证答案
    if "targetWord" in question and "correctAnswer" in question:
        test_answer = "皇帝外出居住的宫殿"
        result = verify_quiz_answer(
            user_answer=test_answer,
            correct_answer=question["correctAnswer"],
            target_word=question["targetWord"]
        )
        print("验证结果：", result)