# punctuate_core.py - 断句核心逻辑

import requests
import json
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::Cn16TxrA"


def punctuate_classic_text(raw_text: str) -> str:
    """
    古文断句函数：每两个字截断，前用逗号，最后用句号
    :param raw_text: 无标点的古文字符串（需去除空格/换行）
    :return: 带标点的断句结果
    """
    # 第一步：清洗输入（去除空格、换行、制表符等无用字符）
    cleaned_text = raw_text.replace(" ", "").replace("\n", "").replace("\t", "")
    if not cleaned_text:  # 空输入处理
        return ""

    try:
        # 2. 调用微调后的GPT-4o-mini模型
        response = openai.chat.completions.create(
            model=FINE_TUNED_MODEL,  # 使用你的微调模型
            messages=[
                {"role": "system", "content": "你是一个专业的古文断句助手，擅长为无标点古文添加准确的传统标点"},
                {"role": "user", "content": cleaned_text}
            ],
            temperature=0,  # 降低随机性，保证断句准确性
            max_tokens=200  # 限制输出长度（根据古文长度调整）
            )

        # 3. 提取模型返回的断句结果
        punctuated_result = response.choices[0].message.content.strip()
        return punctuated_result

    except Exception as e:
        # 捕获API调用异常
        return f"断句失败：{str(e)}"


# 测试代码（可直接运行验证）
if __name__ == "__main__":
    test_input = "学而时习之不亦说乎有朋自远方来不亦乐乎"
    test_output = punctuate_classic_text(test_input)
    print("输入：", test_input)
    print("输出：", test_output)