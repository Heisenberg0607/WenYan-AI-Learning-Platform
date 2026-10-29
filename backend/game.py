import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Set, Tuple
from NER import extract_keywords_with_pos, init_jieba
from RE import RelationExtractor

# BERT相关导入（可选，如果未安装则使用回退方案）
try:
    import torch
    from transformers import BertModel, BertTokenizer
    from sklearn.metrics.pairwise import cosine_similarity
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("⚠ 警告：未安装transformers或torch，将使用基于字符串的相似度计算")
# 获取当前文件 (game.py) 的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 题库 Excel 文件路径 (相对于当前文件)
EXCEL_PATH = os.path.join(BASE_DIR, "output.xlsx")

# 2. 本地 BERT 模型路径 (相对于当前文件)
# 假设 bert-base-chinese 或 bert-ancient-chinese 文件夹就在 backend 目录下
# 如果您使用的是 'bert-ancient-chinese'，请确保文件夹名称一致
LOCAL_BERT_PATH = os.path.join(BASE_DIR, "bert-ancient-chinese")
init_jieba()

class SemanticEntityScorer:
    """基于BERT的语义实体评分器"""
    
    def __init__(self, threshold=0.7, use_bert=True):
        self.threshold = threshold
        self.use_bert = use_bert and BERT_AVAILABLE
        self.model = None
        self.tokenizer = None
        self._entity_cache = {}  # 缓存实体向量，避免重复计算
        
        if self.use_bert:
            try:
                self.model_name = 'bert-base-chinese'
                print(f"📥 正在加载BERT模型: {self.model_name}...")
                self.tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True, cache_dir=None)
                self.model = BertModel.from_pretrained(LOCAL_BERT_PATH, local_files_only=True, cache_dir=None,)
                self.model.eval()  # 设置为评估模式
                print("✅ BERT模型加载成功")
            except Exception as e:
                print(f"⚠ BERT模型加载失败: {e}，将使用回退方案")
                self.use_bert = False
    
    def get_entity_vector(self, entity: str) -> np.ndarray:
        """获取实体的BERT向量表示（使用[CLS] token）"""
        if not self.use_bert:
            # 回退方案：返回简单的one-hot向量（基于字符）
            return np.array([hash(c) % 1000 for c in entity[:10]] + [0] * (10 - len(entity[:10])))
        
        # 检查缓存
        if entity in self._entity_cache:
            return self._entity_cache[entity]
        
        try:
            inputs = self.tokenizer(entity, return_tensors='pt', truncation=True, max_length=10, padding='max_length')
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取[CLS] token的向量作为整个实体的表示
                cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            # 缓存结果
            self._entity_cache[entity] = cls_vector
            return cls_vector
        except Exception as e:
            print(f"⚠ 获取实体向量失败 {entity}: {e}")
            return np.zeros(768)  # BERT-base的隐藏层维度
    
    def calculate_semantic_similarity(self, entity1: str, entity2: str) -> float:
        """计算两个实体之间的语义相似度"""
        if entity1 == entity2:
            return 1.0
        
        vec1 = self.get_entity_vector(entity1)
        vec2 = self.get_entity_vector(entity2)
        
        if self.use_bert:
            # 使用余弦相似度
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            return float(similarity)
        else:
            # 回退方案：基于字符串相似度
            return self._string_similarity(entity1, entity2)
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """字符串相似度（回退方案）"""
        if s1 == s2:
            return 1.0
        if s1 in s2 or s2 in s1:
            return 0.7
        # 计算字符重叠度
        set1 = set(s1)
        set2 = set(s2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def calculate_semantic_overlap(self, user_entities: List[str], correct_entities: List[str]) -> float:
        """计算基于语义相似度的实体匹配度"""
        if not correct_entities:
            return 0.0
        
        if not user_entities:
            return 0.0
        
        # 获取所有实体的向量
        user_vectors = [self.get_entity_vector(ent) for ent in user_entities]
        correct_vectors = [self.get_entity_vector(ent) for ent in correct_entities]
        
        if self.use_bert:
            # 计算相似度矩阵
            sim_matrix = cosine_similarity(user_vectors, correct_vectors)
            
            # 为每个标准实体找到最佳匹配的用户实体
            matched_scores = []
            for j in range(len(correct_entities)):
                max_sim = np.max(sim_matrix[:, j]) if len(user_entities) > 0 else 0
                if max_sim >= self.threshold:
                    matched_scores.append(max_sim)
            
            # 计算平均匹配度
            if matched_scores:
                return float(np.mean(matched_scores))
            else:
                return 0.0
        else:
            # 回退方案：基于字符串相似度
            matched_scores = []
            for correct_ent in correct_entities:
                max_sim = 0.0
                for user_ent in user_entities:
                    sim = self._string_similarity(user_ent, correct_ent)
                    max_sim = max(max_sim, sim)
                if max_sim >= self.threshold:
                    matched_scores.append(max_sim)
            
            return np.mean(matched_scores) if matched_scores else 0.0
    
    def get_matched_pairs(self, user_entities: List[str], correct_entities: List[str]) -> List[Tuple[str, str, float]]:
        """获取匹配的实体对"""
        pairs = []
        for user_ent in user_entities:
            for correct_ent in correct_entities:
                sim = self.calculate_semantic_similarity(user_ent, correct_ent)
                if sim >= self.threshold:
                    pairs.append((user_ent, correct_ent, sim))
        return pairs

class IntelligentScoringSystem:
    """智能评分系统 - 简化版：实体匹配分 + 关系错误扣分"""
    
    def __init__(self, semantic_threshold=0.7, use_bert=True):
        self.re_extractor = RelationExtractor()
        self.semantic_scorer = SemanticEntityScorer(threshold=semantic_threshold, use_bert=use_bert)
        
        # 评分参数
        self.similarity_threshold = semantic_threshold  # 相似度阈值
        self.error_penalty = 15  # 每个错误关系扣15分
        self.correct_threshold = 70  # 正确阈值
        
    def extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体"""
        from NER import extract_word_entities, filter_entities
        import jieba.posseg as pseg
        import re
        
        # 处理"词条：解释"格式
        if '：' in text:
            content = text.split('：', 1)[1] if len(text.split('：')) > 1 else text
        else:
            content = text
        
        # 提取实体
        entities_raw = extract_word_entities(content, set())
        entities = list(filter_entities(entities_raw, content))
        
        # 如果提取不到实体，尝试更宽松的提取方式
        if not entities:
            # 提取所有2-4字的名词性词汇
            words = list(pseg.cut(content))
            stop_words = {'指', '作', '为', '是', '的', '在', '这里', '当时', 
                         '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                         '也', '即', '即指', '称为', '叫做', '。', '，', '；'}
            
            for word, pos in words:
                # 跳过停用词和标点
                if word in stop_words or len(word) < 2:
                    continue
                # 提取名词、形容词、动词（作为备选）
                if pos.startswith(('n', 'a', 'v')) and 2 <= len(word) <= 6:
                    entities.append(word)
            
            # 尝试提取连续的2-4字组合（可能是复合词，如"征雁"）
            # 先去除标点，再提取连续的中文字符组合
            clean_content = re.sub(r'[。，；：、]', '', content)
            # 提取所有连续的中文字符
            chinese_segments = re.findall(r'[\u4e00-\u9fff]+', clean_content)
            
            for segment in chinese_segments:
                # 提取片段中的2-3字子串（滑动窗口），优先提取不含停用词的
                for i in range(len(segment) - 1):
                    for length in [2, 3]:
                        if i + length <= len(segment):
                            sub = segment[i:i+length]
                            # 如果子串不包含停用词，添加
                            if not any(stop in sub for stop in stop_words) and len(sub) >= 2:
                                entities.append(sub)
                
                # 如果整个片段是2-4字，且不包含停用词，也添加
                if 2 <= len(segment) <= 4:
                    if not any(stop in segment for stop in stop_words):
                        entities.append(segment)
        
        # 去重但保持顺序
        unique_entities = []
        seen = set()
        for ent in entities:
            if ent not in seen and len(ent) >= 2:
                unique_entities.append(ent)
                seen.add(ent)
        
        return unique_entities
    
    def calculate_entity_score(self, user_answer: str, correct_answer: str) -> float:
        """计算实体匹配分（完全基于语义相似度，最高100分）"""
        # 提取实体
        user_entities = self.extract_entities(user_answer)
        correct_entities = self.extract_entities(correct_answer)
        
        if not correct_entities:
            return 100.0  # 标准答案无实体，默认满分
        
        if not user_entities:
            return 0.0  # 用户答案无实体，0分
        
        # 使用BERT语义相似度计算匹配度
        semantic_score = self.semantic_scorer.calculate_semantic_overlap(
            user_entities, 
            correct_entities
        )
        
        # 将相似度转换为分数（0-100）
        entity_score = semantic_score * 100
        
        return round(entity_score, 1)
    
    def has_relation_error(self, user_rel: Dict, std_rels: List[Dict]) -> bool:
        """判断关系是否错误（简化逻辑）
        
        错误条件：
        1. 主语相似（相似度 ≥ 阈值）
        2. 关系类型相同
        3. 宾语相似度低（相似度 < 阈值）
        
        注意：如果主语不相似，或者关系类型不同，都不算错误
        """
        user_head = user_rel.get('head', '')
        user_relation = user_rel.get('relation', '')
        user_tail = user_rel.get('tail', '')
        
        for std_rel in std_rels:
            std_head = std_rel.get('head', '')
            std_relation = std_rel.get('relation', '')
            std_tail = std_rel.get('tail', '')
            
            # 1. 关系类型必须相同
            if user_relation != std_relation:
                continue
            
            # 2. 检查主语相似度
            head_similarity = self.semantic_scorer.calculate_semantic_similarity(user_head, std_head)
            if head_similarity < self.similarity_threshold:
                continue  # 主语不相似，不是我们要检查的错误
            
            # 3. 检查宾语相似度
            tail_similarity = self.semantic_scorer.calculate_semantic_similarity(user_tail, std_tail)
            
            # 主语相似、关系相同，但宾语不相似 → 关系错误
            if tail_similarity < self.similarity_threshold:
                return True
        
        return False
    
    def calculate_total_score(self, user_answer: str, correct_answer: str, head_entity: str) -> Dict:
        """计算总分：实体匹配分 - 关系错误扣分"""
        # 1. 计算实体匹配分（基础分）
        entity_score = self.calculate_entity_score(user_answer, correct_answer)
        
        # 2. 提取关系
        user_relations = self.re_extractor.extract_relations(user_answer, head_entity)
        correct_relations = self.re_extractor.extract_relations(correct_answer, head_entity)
        
        # 3. 检查关系错误（只扣分）
        error_count = 0
        error_details = []
        
        if user_relations and correct_relations:
            for user_rel in user_relations:
                if self.has_relation_error(user_rel, correct_relations):
                    error_count += 1
                    error_details.append({
                        'user_relation': user_rel,
                        'error_type': '关系错误'
                    })
        
        # 4. 计算扣分
        penalty = error_count * self.error_penalty
        
        # 5. 计算总分（不能为负数）
        total_score = max(0, entity_score - penalty)
        
        return {
            'total_score': round(total_score, 1),
            'entity_score': entity_score,
            'penalty': penalty,
            'error_count': error_count,
            'error_details': error_details,
            'is_correct': total_score >= self.correct_threshold
        }
    
    def evaluate_answer(self, user_answer: str, correct_answer: str, head_entity: str = None) -> Dict:
        """评估答案（统一接口）"""
        if not head_entity:
            head_entity = correct_answer.split('：')[0].strip() if '：' in correct_answer else ''
        
        # 计算总分
        score_result = self.calculate_total_score(user_answer, correct_answer, head_entity)
        
        # 提取详细信息
        user_entities = self.extract_entities(user_answer)
        correct_entities = self.extract_entities(correct_answer)
        user_relations = self.re_extractor.extract_relations(user_answer, head_entity) if head_entity else []
        correct_relations = self.re_extractor.extract_relations(correct_answer, head_entity) if head_entity else []
        
        # 获取语义匹配的实体对
        matched_pairs = self.semantic_scorer.get_matched_pairs(user_entities, correct_entities)
        
        return {
            'score': score_result['total_score'],
            'is_correct': score_result['is_correct'],
            'entity_score': score_result['entity_score'],
            'penalty': score_result['penalty'],
            'error_count': score_result['error_count'],
            'user_entities': user_entities,
            'correct_entities': correct_entities,
            'user_relations': user_relations,
            'correct_relations': correct_relations,
            'error_relations': [detail['user_relation'] for detail in score_result['error_details']],
            'matched_entity_pairs': matched_pairs,
            'feedback': self.generate_feedback(score_result)
        }
    
    def generate_feedback(self, score_result: Dict) -> str:
        """生成反馈"""
        total_score = score_result['total_score']
        entity_score = score_result['entity_score']
        error_count = score_result['error_count']
        penalty = score_result['penalty']
        
        feedback = []
        
        # 总体评价
        if total_score >= 90:
            feedback.append("✨ 完美！理解非常准确。")
        elif total_score >= 80:
            feedback.append("👍 优秀！抓住了核心含义。")
        elif total_score >= 70:
            feedback.append("✓ 良好！基本理解正确。")
        elif total_score >= 60:
            feedback.append("~ 还行！方向正确但有偏差。")
        else:
            feedback.append("⚠ 需要改进！理解有偏差。")
        
        # 实体反馈
        if entity_score < 60:
            feedback.append("- 遗漏了关键概念或表述不准确。")
        elif entity_score >= 80:
            feedback.append("- 概念提取准确。")
        
        # 关系错误反馈
        if error_count > 0:
            feedback.append(f"- 发现 {error_count} 处关系错误，扣 {penalty} 分。")
        
        
        return "\n".join(feedback)

class PoetryAnnotationGame:
    """简化版古诗词注释游戏"""
    
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.df = None
        self.scorer = IntelligentScoringSystem()
        self.stats = {'total': 0, 'correct': 0, 'total_score': 0}
        
    def load_data(self):
        """加载Excel数据"""
        try:
            self.df = pd.read_excel(self.excel_path, engine='openpyxl')
            print("✅ 数据加载成功")
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def parse_annotation(self, annotation_str):
        """解析注释"""
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
    
    def select_question(self):
        """选择问题"""
        if self.df is None or len(self.df) == 0:
            return None
        
        row = self.df.sample(n=1).iloc[0]
        annotations = self.parse_annotation(row['annotation'])
        
        if not annotations:
            return self.select_question()
        
        # 随机选择一个词条
        term = np.random.choice(list(annotations.keys()))
        
        return {
            'poem_name': str(row['name']).strip(),
            'author': str(row['author']).strip(),
            'content': str(row['content']).strip(),
            'term': term,
            'correct_answer': f"{term}：{annotations[term]}"
        }
    
    def display_question(self, question):
        """显示问题"""
        print("\n" + "═" * 50)
        print(f"📜 《{question['poem_name']}》 - {question['author']}")
        print("═" * 50)
        
        content = question['content']
        term = question['term']
        highlighted = content.replace(term, f"【{term}】")
        
        print(f"\n📝 诗句：{highlighted}")
        print(f"\n❓ 请解释：【{term}】")
        print("-" * 30)
    
    def display_result(self, result, question):
        """显示结果（4部分格式）"""
        user_answer = result.get('user_answer', '')
        correct_answer = question.get('correct_answer', '')
        
        print("\n" + "=" * 60)
        print("📊 评估结果")
        print("=" * 60)
        
        # 第一部分：用户结果与标准答案对比
        print("\n【1. 答案对比】")
        print(f"用户答案：{user_answer}")
        print(f"标准答案：{correct_answer}")
        
        # 第二部分：用户结果与标准答案关键概念列举
        print("\n【2. 关键概念对比】")
        user_entities = result.get('user_entities', [])
        correct_entities = result.get('correct_entities', [])
        
        print(f"用户答案概念 ({len(user_entities)}个):")
        if user_entities:
            for i, ent in enumerate(user_entities[:10], 1):
                print(f"  {i}. {ent}")
            if len(user_entities) > 10:
                print(f"  ... 还有 {len(user_entities) - 10} 个")
        else:
            print("  无")
        
        print(f"\n标准答案概念 ({len(correct_entities)}个):")
        if correct_entities:
            for i, ent in enumerate(correct_entities[:10], 1):
                print(f"  {i}. {ent}")
            if len(correct_entities) > 10:
                print(f"  ... 还有 {len(correct_entities) - 10} 个")
        else:
            print("  无")
        
        # 显示匹配的实体对
        matched_pairs = result.get('matched_entity_pairs', [])
        if matched_pairs:
            print(f"\n语义匹配的实体对 ({len(matched_pairs)}对):")
            for user_ent, correct_ent, sim in matched_pairs[:5]:
                print(f"  {user_ent} <-> {correct_ent} (相似度: {sim:.3f})")
            if len(matched_pairs) > 5:
                print(f"  ... 还有 {len(matched_pairs) - 5} 对")
        
        # 第三部分：用户结果与标准答案关系列举
        print("\n【3. 关系对比】")
        user_relations = result.get('user_relations', [])
        correct_relations = result.get('correct_relations', [])
        
        print(f"用户答案关系 ({len(user_relations)}个):")
        if user_relations:
            for i, rel in enumerate(user_relations[:10], 1):
                head = rel.get('head', '未知')
                relation = rel.get('relation', '未知')
                tail = rel.get('tail', '未知')
                print(f"  {i}. {head} --[{relation}]--> {tail}")
            if len(user_relations) > 10:
                print(f"  ... 还有 {len(user_relations) - 10} 个")
        else:
            print("  无")
        
        print(f"\n标准答案关系 ({len(correct_relations)}个):")
        if correct_relations:
            for i, rel in enumerate(correct_relations[:10], 1):
                head = rel.get('head', '未知')
                relation = rel.get('relation', '未知')
                tail = rel.get('tail', '未知')
                print(f"  {i}. {head} --[{relation}]--> {tail}")
            if len(correct_relations) > 10:
                print(f"  ... 还有 {len(correct_relations) - 10} 个")
        else:
            print("  无")
        
        # 显示错误关系
        error_relations = result.get('error_relations', [])
        if error_relations:
            print(f"\n⚠ 错误关系 ({len(error_relations)}个):")
            for i, rel in enumerate(error_relations[:5], 1):
                head = rel.get('head', '未知')
                relation = rel.get('relation', '未知')
                tail = rel.get('tail', '未知')
                print(f"  {i}. {head} --[{relation}]--> {tail}")
        
        # 第四部分：打分结果和平均分
        print("\n【4. 打分结果】")
        print(f"总分：{result['score']}/100 ({'✅正确' if result['is_correct'] else '❌不正确'})")
        print(f"实体匹配分：{result['entity_score']}/100")
        
        # 显示扣分信息
        penalty = result.get('penalty', 0)
        error_count = result.get('error_count', 0)
        if penalty > 0:
            print(f"关系错误扣分：-{penalty}分（{error_count}处错误）")
        else:
            print(f"关系错误扣分：0分（无错误）")
        
        # 显示平均分
        if self.stats['total'] > 0:
            avg_score = self.stats['total_score'] / self.stats['total']
            accuracy = (self.stats['correct'] / self.stats['total']) * 100
            print(f"\n当前统计：")
            print(f"  答题数：{self.stats['total']}")
            print(f"  正确数：{self.stats['correct']}")
            print(f"  正确率：{accuracy:.1f}%")
            print(f"  平均分：{avg_score:.1f}/100")
        
        # 显示反馈
        print(f"\n💡 反馈：")
        print(result['feedback'])
    
    def play_round(self):
        """进行一轮游戏"""
        question = self.select_question()
        if not question:
            return False
        
        self.display_question(question)
        user_answer = input("\n💭 你的解释：").strip()
        
        if user_answer.lower() in ['quit', '退出', 'q']:
            return False
        
        if not user_answer:
            print("⚠ 请输入答案")
            return True
        
        # 更新统计
        self.stats['total'] += 1
        
        # 评估答案
        result = self.scorer.evaluate_answer(
            user_answer, 
            question['correct_answer'],
            head_entity=question['term']
        )
        
        # 添加用户答案到结果中（用于显示）
        result['user_answer'] = user_answer
        
        if result['is_correct']:
            self.stats['correct'] += 1
        self.stats['total_score'] += result['score']
        
        # 显示结果
        self.display_result(result, question)
        
        # 显示统计
        if self.stats['total'] > 0:
            accuracy = (self.stats['correct'] / self.stats['total']) * 100
            avg_score = self.stats['total_score'] / self.stats['total']
            print(f"\n📈 当前：第{self.stats['total']}题 | 正确率：{accuracy:.1f}% | 平均分：{avg_score:.1f}")
        
        return True
    
    def run(self):
        """运行游戏"""
        print("\n" + "☆" * 50)
        print("🎮 古诗词注释理解游戏")
        print("☆" * 50)
        print("规则：解释诗中【高亮】词语 | 输入 quit/退出/q 可随时退出")
        
        if not self.load_data():
            return
        
        input("\n按回车键开始...")
        
        while True:
            if not self.play_round():
                break
            
            cont = input("\n继续下一题？(y/n): ").lower().strip()
            if cont not in ['y', 'yes', '是', '继续', '']:
                break
        
        # 最终统计
        if self.stats['total'] > 0:
            print("\n" + "═" * 50)
            print("🏆 游戏结束 - 最终成绩")
            print("═" * 50)
            
            accuracy = (self.stats['correct'] / self.stats['total']) * 100
            avg_score = self.stats['total_score'] / self.stats['total']
            
            print(f"\n📊 统计：")
            print(f"   答题数：{self.stats['total']}")
            print(f"   正确数：{self.stats['correct']}")
            print(f"   正确率：{accuracy:.1f}%")
            print(f"   平均分：{avg_score:.1f}/100")
            
            # 评级
            if avg_score >= 90:
                print("\n🥇 金牌学者 - 古诗词达人！")
            elif avg_score >= 80:
                print("\n🥈 银牌学者 - 理解能力优秀！")
            elif avg_score >= 70:
                print("\n🥉 铜牌学者 - 掌握良好！")
            elif avg_score >= 60:
                print("\n📚 合格学者 - 继续努力！")
            else:
                print("\n📖 潜力学者 - 多加练习！")

# 主程序
if __name__ == "__main__":
    if not os.path.exists(EXCEL_PATH):
        print(f"❌ 找不到数据文件: {EXCEL_PATH}")
    else:
        game = PoetryAnnotationGame(EXCEL_PATH)
        game.run()