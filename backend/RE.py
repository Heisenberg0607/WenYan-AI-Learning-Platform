# RE.py - 基于规则的全面关系抽取器
import re
from typing import List, Dict, Set
from NER import extract_word_entities, filter_entities, init_jieba

class RelationExtractor:
    """基于规则的全面关系抽取器"""
    
    def __init__(self, use_bert: bool = False):
        init_jieba()
        self.use_bert = use_bert
        
        # 定义全面的关系抽取规则（按优先级排序）
        self.relation_patterns = {
            '特指': [
                # 这里指/特指/指的是/即指/指（最高优先级）
                (r'(?:这里指|特指|指的是|即指|指)([^。，；]{2,15}?)(?:。|，|；|$)', 0.95),
            ],
            '同义': [
                # 一作/又称/亦称/也称
                (r'一作[《「]?([^」》。]{1,10})[」》]?', 0.95),
                (r'又称[《「]?([^」》。]{1,10})[」》]?', 0.95),
                (r'亦称[《「]?([^」》。]{1,10})[」》]?', 0.95),
                (r'也称([^。，；]{1,10})', 0.90),
                (r'也叫([^。，；]{1,10})', 0.90),
                # 或...（同义）
                (r'或([^，。；]{1,10})(?:。|，|；|$)', 0.80),
            ],
            '引用': [
                # 出自/来自/引自...（优先于其他模式）
                (r'出自[《「]?([^」》。]{2,20})[」》]?', 0.95),
                (r'来自[《「]?([^」》。]{2,20})[」》]?', 0.90),
                (r'引自[《「]?([^」》。]{2,20})[」》]?', 0.95),
                (r'见[《「]?([^」》。]{2,20})[」》]?', 0.85),
                # 书名号（但避免与"一作《X》"冲突）
                (r'(?<!一作)(?<!又称)(?<!亦称)[《「]([^」》]{2,20})[」》]', 0.80),
            ],
            '地理位置': [
                # 位于/旧址在/前对/下临（明确的位置关系）
                (r'位于([^，。；]{2,15}?)(?:，|。|；|$)', 0.95),
                (r'旧址在([^，。；]{2,15}?)(?:，|。|；|$)', 0.95),
                (r'前对([^，。；]{2,12}?)(?:，|。|；|$)', 0.90),
                (r'下临([^，。；]{2,12}?)(?:，|。|；|$)', 0.90),
                # 在...之X（位置关系，优先匹配完整的"X之Y"）
                (r'在([^之]{1,10})之(外|内|上|下|前|后|东|西|南|北)', 0.90),
            ],
            '所属关系': [
                # X的Y（所属关系，但要避免提取"X在Y的Z"这种结构）
                (r'(?<!在)(?<!位于)([^，。；]{1,8}?)的([^，。；]{1,8}?)(?:。|，|；|$)', 0.85),
                # 属于.../归于...
                (r'属于([^，。；]{2,12}?)(?:。|，|；|$)', 0.90),
                (r'归于([^，。；]{2,12}?)(?:。|，|；|$)', 0.85),
            ],
            '定义': [
                # 是.../为...（定义关系，但要避免与"属于"冲突）
                (r'(?<!属于)(?<!归于)是([^，。；]{2,20}?)(?:。|，|；|$)', 0.90),
                (r'为([^，。；]{2,20}?)(?:。|，|；|$)', 0.85),
                # 表示.../意思是...
                (r'表示([^，。；]{2,20}?)(?:。|，|；|$)', 0.80),
                (r'意思是([^，。；]{2,20}?)(?:。|，|；|$)', 0.80),
            ],
            '属性': [
                # 形容.../描述...
                (r'形容([^，。；]{2,12}?)(?:。|，|；|$)', 0.85),
                (r'描述([^，。；]{2,12}?)(?:。|，|；|$)', 0.80),
                # ...颜色/...形状等
                (r'([^，。；]{1,6})(?:色|颜色|形状|大小|特征)', 0.75),
                # 鲜红色/深蓝色等
                (r'([^，。；]{1,4})(?:红|绿|蓝|黄|黑|白|紫)色', 0.80),
            ],
            '动作': [
                # 做/进行/执行...（动作关系，优先级较低）
                (r'做([^，。；]{2,12}?)(?:。|，|；|$)', 0.75),
                (r'进行([^，。；]{2,12}?)(?:。|，|；|$)', 0.80),
                (r'执行([^，。；]{2,12}?)(?:。|，|；|$)', 0.80),
            ]
        }
    
    def extract_relations(self, text: str, head_entity: str) -> List[Dict]:
        """
        抽取注释中的所有关系
        
        参数:
            text: 注释文本（可以是"词条：解释"格式，或直接是解释）
            head_entity: 目标词条
        
        返回:
            关系三元组列表
        """
        relations = []
        
        if not text or not head_entity:
            return relations
        
        # 提取解释部分
        explanation = self._extract_explanation(text, head_entity)
        if not explanation:
            return relations
        
        # 提取实体列表（用于验证关系中的实体）
        entities = self._extract_entities_from_text(explanation, head_entity)
        
        # 清理文本
        clean_text = self._clean_text(explanation)
        
        # 遍历所有关系模式
        for rel_type, patterns in self.relation_patterns.items():
            for pattern, base_confidence in patterns:
                matches = re.finditer(pattern, clean_text)
                for match in matches:
                    tail = self._extract_tail_from_match(match, rel_type)
                    if tail:
                        # 清理tail
                        tail = self._clean_entity(tail)
                        
                        # 验证tail是否在实体列表中（或至少部分匹配）
                        if self._is_valid_entity(tail, entities, head_entity):
                            relations.append({
                                'head': head_entity,
                                'relation': rel_type,
                                'tail': tail,
                                'confidence': self._calculate_confidence(match, clean_text, base_confidence),
                                'evidence': match.group()
                            })
        
        # 特殊处理：定义关系（如果整个解释就是定义，且没有其他明确关系）
        if not any(r['relation'] == '定义' for r in relations):
            # 如果解释很短且没有其他明确关系，可能是定义
            has_explicit_relation = any(r['relation'] in ['特指', '同义', '引用', '地理位置'] for r in relations)
            if not has_explicit_relation and len(explanation) < 50:
                # 提取解释中的关键实体作为tail（优先选择较短的实体）
                key_entities = sorted([e for e in entities if e != head_entity and len(e) >= 2], 
                                     key=lambda x: (len(x), x))[:2]
                for entity in key_entities:
                    if entity in explanation and len(entity) <= 8:
                        relations.append({
                            'head': head_entity,
                            'relation': '定义',
                            'tail': entity,
                            'confidence': 0.70,
                            'evidence': explanation
                        })
        
        # 去重并排序
        relations = self._deduplicate_relations(relations)
        
        # 过滤低质量关系
        relations = self._filter_low_quality_relations(relations, head_entity, explanation)
        
        # 按置信度排序
        relations = sorted(relations, key=lambda x: x['confidence'], reverse=True)
        
        return relations
    
    def _extract_explanation(self, text: str, head_entity: str) -> str:
        """从文本中提取解释部分"""
        if '：' in text:
            parts = text.split('：', 1)
            if len(parts) == 2:
                return parts[1].strip()
        elif ':' in text:
            parts = text.split(':', 1)
            if len(parts) == 2:
                return parts[1].strip()
        
        # 如果没有分隔符，直接返回文本（可能是纯解释）
        return text.strip()
    
    def _extract_entities_from_text(self, explanation: str, head_entity: str) -> Set[str]:
        """从解释中提取实体"""
        entities = set([head_entity])  # 确保head_entity在实体列表中
        
        # 使用NER模块提取实体
        try:
            word_entities = extract_word_entities(explanation, set())
            filtered = filter_entities(word_entities, explanation)
            entities.update(filtered)
        except:
            # 如果NER模块失败，使用简单方法
            pass
        
        return entities
    
    def _extract_tail_from_match(self, match, rel_type: str) -> str:
        """从匹配中提取尾部实体"""
        groups = match.groups()
        if not groups:
            return ""
        
        # 根据关系类型处理
        if rel_type == '地理位置' and len(groups) >= 2:
            # 对于"在...之X"模式，组合成"XXX之X"
            if groups[1] in ['外', '内', '上', '下', '前', '后', '东', '西', '南', '北']:
                return f"{groups[0]}之{groups[1]}"
            else:
                return groups[0] if groups[0] else ""
        elif rel_type == '所属关系' and len(groups) >= 2:
            # 对于"X的Y"模式，返回Y（被拥有者）
            return groups[1] if groups[1] else groups[0]
        else:
            # 取第一个非空组
            for group in groups:
                if group and isinstance(group, str):
                    return group.strip()
        
        return ""
    
    def _clean_entity(self, entity: str) -> str:
        """清理实体"""
        if not entity:
            return ""
        
        # 去除引号、书名号等
        entity = re.sub(r'[《》「」""\'\'「」""]', '', entity)
        # 去除末尾标点
        entity = re.sub(r'[，。；：,.!?;:\s]$', '', entity)
        # 去除开头标点
        entity = re.sub(r'^[，。；：,.!?;:\s]+', '', entity)
        
        return entity.strip()
    
    def _is_valid_entity(self, tail: str, entities: Set[str], head_entity: str) -> bool:
        """验证tail是否是有效实体"""
        if not tail or tail == head_entity:
            return False
        
        # 直接匹配
        if tail in entities:
            return True
        
        # 部分匹配（tail是某个实体的子串，或某个实体是tail的子串）
        for entity in entities:
            if tail in entity or entity in tail:
                # 但避免太短的匹配
                if min(len(tail), len(entity)) >= 2:
                    return True
        
        # 如果tail长度合理且包含中文字符，也接受
        if 2 <= len(tail) <= 15 and any('\u4e00' <= char <= '\u9fff' for char in tail):
            return True
        
        return False
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 去除特殊字符但保留中文标点
        text = re.sub(r'[▲◆◇■□●○★☆]', '', text)
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _calculate_confidence(self, match, text: str, base_confidence: float) -> float:
        """计算置信度"""
        confidence = base_confidence
        
        # 匹配长度越长，置信度越高
        match_text = match.group()
        if len(match_text) >= 6:
            confidence += 0.05
        
        # 如果匹配在文本开头，置信度更高
        if match.start() < len(text) * 0.3:  # 在前30%
            confidence += 0.05
        
        # 包含特定关键词
        high_confidence_keywords = ['指', '即', '一作', '位于', '出自']
        if any(keyword in match_text for keyword in high_confidence_keywords):
            confidence += 0.05
        
        return min(0.98, confidence)  # 上限0.98
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """关系去重"""
        seen = set()
        unique = []
        
        for rel in relations:
            # 使用(head, relation, tail)作为唯一键
            key = (rel['head'], rel['relation'], rel['tail'])
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    
    def _filter_low_quality_relations(self, relations: List[Dict], head_entity: str, explanation: str) -> List[Dict]:
        """过滤低质量关系"""
        filtered = []
        
        # 合并"地理位置"和"位于"（保留置信度更高的）
        geo_relations = {}
        other_relations = []
        
        for rel in relations:
            if rel['relation'] in ['地理位置', '位于']:
                tail = rel['tail']
                if tail not in geo_relations or rel['confidence'] > geo_relations[tail]['confidence']:
                    # 统一使用"地理位置"关系类型
                    geo_relations[tail] = {
                        'head': rel['head'],
                        'relation': '地理位置',
                        'tail': tail,
                        'confidence': rel['confidence'],
                        'evidence': rel.get('evidence', '')
                    }
            else:
                other_relations.append(rel)
        
        # 处理地理位置关系：如果同时有"X"和"X之Y"，只保留"X之Y"
        geo_tails = list(geo_relations.keys())
        for tail1 in geo_tails:
            for tail2 in geo_tails:
                if tail1 != tail2 and tail1 in tail2 and tail2.endswith(('之外', '之内', '之上', '之下', '之前', '之后', '之东', '之西', '之南', '之北')):
                    # tail1是tail2的子串，且tail2是完整的"X之Y"格式，删除tail1
                    if tail1 in geo_relations:
                        del geo_relations[tail1]
        
        # 添加地理位置关系
        filtered.extend(geo_relations.values())
        
        # 过滤其他关系
        for rel in other_relations:
            # 过滤掉tail太长的关系（可能是误提取）
            if len(rel['tail']) > 20:
                continue
            
            # 过滤掉tail包含head的关系（避免循环，但允许部分匹配）
            if rel['head'] in rel['tail'] and len(rel['tail']) > len(rel['head']) + 3:
                # 但如果tail是"X的Y"格式，且head是X，则保留
                if '的' not in rel['tail']:
                    continue
            
            # 过滤掉tail是常见停用词的关系
            stop_words = {'的', '了', '在', '是', '有', '和', '与', '或', '及', '等', '指', '即', '作', '这里', '当时'}
            if rel['tail'] in stop_words or len(rel['tail']) < 2:
                continue
            
            # 过滤掉tail只包含标点或数字的关系
            if not any('\u4e00' <= char <= '\u9fff' for char in rel['tail']):
                continue
            
            filtered.append(rel)
        
        return filtered
