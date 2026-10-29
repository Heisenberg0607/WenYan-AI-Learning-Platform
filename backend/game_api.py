"""
古诗词注释评分系统 - 函数式API接口
用于前端接入的简化接口
"""

from typing import Dict, List, Optional
from game import IntelligentScoringSystem, PoetryAnnotationGame
import pandas as pd
import os

# 全局评分系统实例（单例模式，避免重复加载BERT模型）
_scorer = None
_game = None

def init_scorer(semantic_threshold: float = 0.7, use_bert: bool = True) -> None:
    """
    初始化评分系统（全局单例）
    
    参数:
        semantic_threshold: 语义相似度阈值，默认0.7
        use_bert: 是否使用BERT模型，默认True
    """
    global _scorer
    if _scorer is None:
        _scorer = IntelligentScoringSystem(
            semantic_threshold=semantic_threshold,
            use_bert=use_bert
        )
    return _scorer

def init_game(excel_path: str = "output.xlsx") -> bool:
    """
    初始化游戏数据（全局单例）
    
    参数:
        excel_path: Excel数据文件路径，默认"output.xlsx"
    
    返回:
        bool: 是否初始化成功
    """
    global _game
    if _game is None:
        if not os.path.exists(excel_path):
            return False
        _game = PoetryAnnotationGame(excel_path)
        return _game.load_data()
    return True

def evaluate_answer(
    user_answer: str,
    correct_answer: str,
    head_entity: Optional[str] = None,
    semantic_threshold: float = 0.7,
    use_bert: bool = True
) -> Dict:
    """
    评估用户答案（核心评分函数）
    
    参数:
        user_answer: 用户的答案文本
        correct_answer: 标准答案文本（格式："词条：解释"）
        head_entity: 词条（如果correct_answer中没有"："，需要提供）
        semantic_threshold: 语义相似度阈值，默认0.7
        use_bert: 是否使用BERT模型，默认True
    
    返回:
        Dict: 评分结果，包含以下字段：
            - score: 总分 (0-100)
            - is_correct: 是否判定为正确 (bool)
            - entity_score: 实体匹配分 (0-100)
            - penalty: 关系错误扣分 (0+)
            - error_count: 错误关系数量
            - user_entities: 用户答案提取的实体列表
            - correct_entities: 标准答案提取的实体列表
            - user_relations: 用户答案提取的关系列表
            - correct_relations: 标准答案提取的关系列表
            - error_relations: 错误的关系列表
            - matched_entity_pairs: 语义匹配的实体对列表 [(user_ent, correct_ent, similarity), ...]
            - feedback: 反馈文本
    
    示例:
        >>> result = evaluate_answer(
        ...     user_answer="皇帝外出居住的宫殿",
        ...     correct_answer="行宫：皇帝在京城之外的宫殿。这里指当时东都洛阳的皇帝行宫上阳宫。",
        ...     head_entity="行宫"
        ... )
        >>> print(f"得分: {result['score']}/100")
    """
    scorer = init_scorer(semantic_threshold, use_bert)
    return scorer.evaluate_answer(user_answer, correct_answer, head_entity)

def extract_entities(text: str) -> List[str]:
    """
    从文本中提取实体
    
    参数:
        text: 输入文本（可以是"词条：解释"格式或纯文本）
    
    返回:
        List[str]: 提取的实体列表
    
    示例:
        >>> entities = extract_entities("行宫：皇帝在京城之外的宫殿")
        >>> print(entities)  # ['皇帝', '宫殿', '京城之外']
    """
    scorer = init_scorer()
    return scorer.extract_entities(text)

def get_random_question(excel_path: str = "output.xlsx") -> Optional[Dict]:
    """
    随机获取一道题目
    
    参数:
        excel_path: Excel数据文件路径，默认"output.xlsx"
    
    返回:
        Dict: 题目信息，包含以下字段：
            - poem_name: 诗词名称
            - author: 作者
            - content: 诗词内容
            - term: 词条
            - correct_answer: 标准答案（格式："词条：解释"）
        如果获取失败，返回None
    
    示例:
        >>> question = get_random_question()
        >>> if question:
        ...     print(f"请解释：{question['term']}")
        ...     print(f"诗词：{question['content']}")
    """
    if not init_game(excel_path):
        return None
    
    global _game
    return _game.select_question()

def parse_annotation(annotation_str: str) -> Dict[str, str]:
    """
    解析注释字符串（支持<br>分隔的多条注释）
    
    参数:
        annotation_str: 注释字符串，格式："词条1：解释1<br>词条2：解释2"
    
    返回:
        Dict[str, str]: 词条到解释的映射字典
    
    示例:
        >>> annotations = parse_annotation("行宫：皇帝在京城之外的宫殿<br>寥落：寂寞冷落")
        >>> print(annotations)  # {'行宫': '皇帝在京城之外的宫殿', '寥落': '寂寞冷落'}
    """
    if not isinstance(annotation_str, str):
        return {}
    
    annotation_dict = {}
    for part in annotation_str.strip().split('<br>'):
        part = part.strip()
        if part and not part.startswith('▲'):
            if '：' in part:
                key, value = part.split('：', 1)
                annotation_dict[key.strip()] = value.strip()
    
    return annotation_dict

def batch_evaluate(
    user_answers: List[str],
    correct_answers: List[str],
    head_entities: Optional[List[str]] = None,
    semantic_threshold: float = 0.7,
    use_bert: bool = True
) -> List[Dict]:
    """
    批量评估多个答案
    
    参数:
        user_answers: 用户答案列表
        correct_answers: 标准答案列表
        head_entities: 词条列表（可选，如果correct_answers中没有"："，需要提供）
        semantic_threshold: 语义相似度阈值，默认0.7
        use_bert: 是否使用BERT模型，默认True
    
    返回:
        List[Dict]: 评分结果列表，每个元素格式同evaluate_answer的返回值
    
    示例:
        >>> results = batch_evaluate(
        ...     user_answers=["皇帝外出居住的宫殿", "寂寞冷清"],
        ...     correct_answers=[
        ...         "行宫：皇帝在京城之外的宫殿",
        ...         "寥落：寂寞冷落"
        ...     ],
        ...     head_entities=["行宫", "寥落"]
        ... )
        >>> for i, result in enumerate(results):
        ...     print(f"第{i+1}题得分: {result['score']}/100")
    """
    scorer = init_scorer(semantic_threshold, use_bert)
    results = []
    
    for i, (user_answer, correct_answer) in enumerate(zip(user_answers, correct_answers)):
        head_entity = head_entities[i] if head_entities and i < len(head_entities) else None
        result = scorer.evaluate_answer(user_answer, correct_answer, head_entity)
        results.append(result)
    
    return results

# 便捷函数：格式化输出结果
def format_result(result: Dict) -> str:
    """
    格式化评分结果为可读字符串
    
    参数:
        result: evaluate_answer返回的结果字典
    
    返回:
        str: 格式化后的字符串
    """
    lines = [
        f"总分: {result['score']}/100 ({'✅正确' if result['is_correct'] else '❌不正确'})",
        f"实体匹配分: {result['entity_score']}/100",
        f"关系错误扣分: -{result['penalty']}分 ({result['error_count']}处错误)",
        "",
        f"用户答案实体 ({len(result['user_entities'])}个): {', '.join(result['user_entities'][:10])}",
        f"标准答案实体 ({len(result['correct_entities'])}个): {', '.join(result['correct_entities'][:10])}",
        "",
        f"反馈: {result['feedback']}"
    ]
    return "\n".join(lines)


# 示例使用
if __name__ == "__main__":
    print("=" * 60)
    print("古诗词注释评分系统 - API使用示例")
    print("=" * 60)
    
    # 示例1: 单次评分
    print("\n【示例1: 单次评分】")
    result = evaluate_answer(
        user_answer="皇帝外出居住的宫殿",
        correct_answer="行宫：皇帝在京城之外的宫殿。这里指当时东都洛阳的皇帝行宫上阳宫。",
        head_entity="行宫"
    )
    print(format_result(result))
    
    # 示例2: 提取实体
    print("\n【示例2: 提取实体】")
    entities = extract_entities("行宫：皇帝在京城之外的宫殿")
    print(f"提取的实体: {entities}")
    
    # 示例3: 获取随机题目
    print("\n【示例3: 获取随机题目】")
    question = get_random_question()
    if question:
        print(f"诗词: {question['poem_name']} - {question['author']}")
        print(f"内容: {question['content']}")
        print(f"词条: {question['term']}")
        print(f"标准答案: {question['correct_answer']}")
    
    # 示例4: 批量评估
    print("\n【示例4: 批量评估】")
    results = batch_evaluate(
        user_answers=["皇帝外出居住的宫殿", "寂寞冷清"],
        correct_answers=[
            "行宫：皇帝在京城之外的宫殿",
            "寥落：寂寞冷落"
        ],
        head_entities=["行宫", "寥落"]
    )
    for i, result in enumerate(results, 1):
        print(f"\n第{i}题: {result['score']}/100")

