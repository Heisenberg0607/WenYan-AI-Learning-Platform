# NER.py - 针对关系抽取优化的版本 (修正版V2)
import re
import jieba
import jieba.posseg as pseg
from typing import List, Tuple, Dict, Set

def init_jieba():
    """专门针对古诗词注释的词典初始化"""
    # 专有名词和复合词库
    domain_entities = [
        '行宫', '上阳宫', '东都洛阳', '京城之外', '鹳雀楼', '中条山', '永济市',
        '小姑子', '相思子', '李龟年', '贺知章', '白居易', '唐玄宗', '黄鹤楼',
        '白发宫人', '剑南文学', '乐游原', '未央宫', '受降城', '洞庭湖',
        '皇宫', '冷宫', '黄河', '长江', '江南', '湘潭', '烟瘴', '风物'
    ]
    
    for word in domain_entities:
        jieba.add_word(word, freq=10000, tag='nz')

def extract_entities_for_relation(text: str) -> Dict[str, Set[str]]:
    """
    为关系抽取优化的实体提取
    返回: {'entities': set(实体)}
    """
    if not isinstance(text, str) or '：' not in text:
        return {'entities': set()}
    
    _, explanation = text.split('：', 1)
    explanation = explanation.strip('。')
    
    # 提取独立实体
    word_entities = extract_word_entities(explanation, set())
    
    # 过滤实体
    filtered_entities = filter_entities(word_entities, explanation)
    
    return {
        'entities': filtered_entities
    }

def extract_key_phrases(explanation: str) -> Set[str]:
    """精确提取关键短语"""
    phrases = set()
    
    # 停用词（短语不应以这些词开头或结尾）
    stop_words = {'指', '作', '为', '是', '的', '在', '之', '这里', '当时', '也', '即'}
    
    # 特指关系：这里指/特指/指的是/指
    ref_patterns = [
        r'这里指([^。，；]{2,12}?)(?:。|，|；|$)',
        r'特指([^。，；]{2,12}?)(?:。|，|；|$)',
        r'指的是([^。，；]{2,12}?)(?:。|，|；|$)',
        r'指([^。，；]{2,12}?)(?:的|之|等)?(?:。|，|；|$)'
    ]
    
    for pattern in ref_patterns:
        matches = re.findall(pattern, explanation)
        for match in matches:
            clean_phrase = clean_entity(match)
            # 过滤掉以停用词开头或结尾的短语
            if clean_phrase and not (clean_phrase.startswith(tuple(stop_words)) or clean_phrase.endswith(tuple(stop_words))):
                phrases.add(clean_phrase)
    
    # 同义关系：一作/又称/亦称/也称
    syn_patterns = [
        r'一作[《「]?([^」》。]{1,8})[」》]?',
        r'又称[《「]?([^」》。]{1,8})[」》]?',
        r'亦称[《「]?([^」》。]{1,8})[」》]?',
        r'也称([^。，；]{1,8})'
    ]
    
    for pattern in syn_patterns:
        matches = re.findall(pattern, explanation)
        for match in matches:
            clean_phrase = clean_entity(match)
            if clean_phrase:
                phrases.add(clean_phrase)
    
    # 位置关系：在...之X/位于/旧址在/前对/下临
    loc_patterns = [
        r'在([^之]{1,10})之(外|内|上|下|前|后|东|西|南|北)',
        r'位于([^，。]{2,15}?)(?:，|。|$)',
        r'旧址在([^，。]{2,15}?)(?:，|。|$)',
        r'前对([^，。]{2,12}?)(?:，|。|$)',
        r'下临([^，。]{2,12}?)(?:，|。|$)'
    ]
    
    for pattern in loc_patterns:
        matches = re.findall(pattern, explanation)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 2:
                    # 对于"在...之X"模式，组合成"XXX之X"
                    phrase = f"{match[0]}之{match[1]}"
                else:
                    phrase = match[0]
            else:
                phrase = match
                
            clean_phrase = clean_entity(phrase)
            # 过滤掉以停用词开头或结尾的短语，以及过长的短语
            if clean_phrase and len(clean_phrase) <= 8 and not (clean_phrase.startswith(tuple(stop_words)) or clean_phrase.endswith(tuple(stop_words))):
                phrases.add(clean_phrase)
    
    # 所属关系：X的Y
    pos_matches = re.findall(r'([^，。]{1,6})的([^，。]{1,6})', explanation)
    for owner, possession in pos_matches:
        clean_owner = clean_entity(owner)
        clean_possession = clean_entity(possession)
        if clean_owner:
            phrases.add(clean_owner)
        if clean_possession:
            phrases.add(clean_possession)
    
    return phrases

def extract_word_entities(explanation: str, existing_phrases: Set[str]) -> Set[str]:
    """提取独立实体"""
    entities = set()
    
    # 分词
    words = list(pseg.cut(explanation))
    
    # 明确的停用词
    stop_words = {'指', '作', '为', '是', '的', '在', '这里', '当时', 
                  '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                  '也', '即', '即指', '称为', '叫做'}
    
    # 优先提取名词、地名、人名等实体
    for word, pos in words:
        # 过滤停用词和短词
        if word in stop_words or len(word) < 2:
            continue
            
        # 优先保留名词、地名、机构名等实体
        if pos.startswith(('n', 'ns', 'nr', 'nt', 'nz')):
            entities.add(word)
    
    # 尝试构建2-3词的复合实体
    compound_entities = build_compound_entities(words, existing_phrases)
    entities.update(compound_entities)
    
    # 如果没有提取到任何实体，放宽条件：也提取形容词和动词（作为备选）
    if not entities:
        for word, pos in words:
            # 过滤停用词和短词
            if word in stop_words or len(word) < 2:
                continue
            
            # 允许形容词（a开头）和部分动词（v开头，但排除常见动词）
            if pos.startswith('a') or (pos.startswith('v') and word not in {'是', '有', '在', '为', '作'}):
                # 确保词在原文中出现
                if word in explanation:
                    entities.add(word)
    
    return entities

def build_compound_entities(words: List, existing_phrases: Set[str]) -> Set[str]:
    """构建合理的复合实体"""
    compounds = set()
    
    # 复合实体专用的停用词（比独立实体过滤宽松，保留'之'）
    compound_stop_words = {'指', '作', '为', '是', '的', '在', '这里', '当时', 
                           '也', '即', '即指', '称为', '叫做'}
    
    i = 0
    while i < len(words):
        word, pos = words[i]
        
        # 如果是名词性词语，尝试向后合并
        if pos.startswith(('n', 'ns', 'nr', 'nt', 'nz')):
            # 构建2-3词的复合实体
            for length in [2, 3]:
                if i + length <= len(words):
                    compound_words = words[i:i+length]
                    # 解包所有pair对象
                    word_pos_pairs = []
                    for w in compound_words:
                        if isinstance(w, tuple):
                            word_pos_pairs.append(w)
                        else:
                            # 处理pair对象
                            word_pos_pairs.append((w.word, w.flag))
                    
                    # 检查每个词：只允许名词性词语和'之'，拒绝动词和其他停用词
                    valid_words = []
                    contains_stop_word = False
                    
                    for p in word_pos_pairs:
                        w, p_pos = p[0], p[1]
                        # 如果是停用词（除了'之'），标记并跳过这个复合实体
                        if w in compound_stop_words:
                            contains_stop_word = True
                            break
                        # 只允许名词性词语和'之'，拒绝动词、形容词等
                        if p_pos.startswith(('n', 'ns', 'nr', 'nt', 'nz')) or w == '之':
                            valid_words.append(w)
                        # 其他词性（如动词、形容词）不允许加入复合实体
                        else:
                            contains_stop_word = True
                            break
                    
                    # 如果包含停用词，跳过
                    if contains_stop_word or not valid_words:
                        continue
                    
                    compound = ''.join(valid_words)
                    
                    # 再次检查：确保复合实体本身不包含停用词（除了'之'）
                    # 检查是否包含停用词作为子串
                    if any(stop in compound and stop != '之' for stop in compound_stop_words):
                        continue
                    
                    # 检查是否在原文中合理出现
                    if 2 <= len(compound) <= 8:
                        compounds.add(compound)
            
            # 跳过已处理的词
            i += 1
        else:
            i += 1
    
    return compounds

def filter_entities(entities: Set[str], explanation: str) -> Set[str]:
    """过滤实体，确保不包含标点符号和停用词"""
    filtered = set()
    
    # 停用词列表（实体中不应包含这些词，但保留'之'）
    stop_words_in_entity = {'指', '作', '为', '是', '的', '在', '这里', '当时', 
                            '也', '即', '即指', '称为', '叫做', '一', '二', '三'}
    
    # 标点符号正则表达式
    punctuation_pattern = re.compile(r'[，。；：,.!?;:\s、""''（）()【】\\[\\]《》]')
    
    for entity in entities:
        # 先清理实体，去除标点符号
        entity = clean_entity(entity)
        
        # 长度过滤
        if len(entity) < 2 or len(entity) > 10:
            continue
        
        # 再次检查是否包含标点符号（双重保险）
        if punctuation_pattern.search(entity):
            continue
        
        # 检查实体是否以停用词开头或结尾（除了'之'）
        if entity.startswith(tuple(stop_words_in_entity)) or entity.endswith(tuple(stop_words_in_entity)):
            continue
        
        # 检查实体中间是否包含停用词（除了'之'）
        # 如果实体包含'在'等停用词，过滤掉（但允许'之'）
        contains_stop = False
        for stop in stop_words_in_entity:
            if stop in entity:
                contains_stop = True
                break
        if contains_stop:
            continue
            
        # 检查是否在原文中出现（允许部分匹配）
        if entity in explanation:
            filtered.add(entity)
        else:
            # 对于复合实体，检查组成部分是否都在原文中
            all_parts_in_text = True
            for char in entity:
                if char not in explanation:
                    all_parts_in_text = False
                    break
            
            if all_parts_in_text:
                filtered.add(entity)
    
    # 去除相互包含的实体（保留更长的）
    final_entities = set()
    entities_list = sorted(filtered, key=len, reverse=True)
    
    for i, entity in enumerate(entities_list):
        is_contained = False
        for j in range(i):
            if entity in entities_list[j] and entity != entities_list[j]:
                is_contained = True
                break
        
        if not is_contained:
            final_entities.add(entity)
    
    return final_entities

def clean_entity(entity: str) -> str:
    """清理实体，去除所有标点符号"""
    # 去除引号、书名号等
    entity = re.sub(r'[《》「」""\'\'「」""]', '', entity)
    # 去除所有标点符号（包括中文和英文标点）
    entity = re.sub(r'[，。；：,.!?;:\s、""''（）()【】\\[\\]《》]', '', entity)
    return entity.strip()

def extract_keywords_with_pos(text: str, head_entity: str = None) -> List[Tuple[str, str]]:
    """保持原有接口的兼容函数"""
    result = extract_entities_for_relation(text)
    entities_list = list(result['entities'])
    return [(entity, 'n') for entity in entities_list]

# 测试函数
if __name__ == "__main__":
    init_jieba()
    
    test_texts = [
        '行宫：皇帝在京城之外的宫殿。这里指当时东都洛阳的皇帝行宫上阳宫。',
        '小姑：一作"小娘"。丈夫的妹妹。也称小姑子。',
        '鹳雀楼：旧址在山西永济市，前对中条山，下临黄河。',
        '寥落：寂寞冷落。',
        '玄宗：指唐玄宗。'
    ]
    
    for text in test_texts:
        print(f"\n原文: {text}")
        result = extract_entities_for_relation(text)
        print(f"提取的实体: {result['entities']}")