import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

TRANSLATE_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::CoQEbRfm"


def translate_classic_text(raw_text: str) -> str:
    """
    古文翻译函数：将古文翻译成现代文
    :param raw_text: 需要翻译的古文字符串
    :return: 翻译后的现代文结果
    """
    # 清洗输入（去除空格、换行、制表符等无用字符）
    cleaned_text = raw_text.replace(" ", "").replace("\n", "").replace("\t", "")
    if not cleaned_text:  # 空输入处理
        return ""

    try:
        # 调用翻译微调模型
        response = openai.chat.completions.create(
            model=TRANSLATE_MODEL,  # 使用翻译微调模型
            messages=[
                {"role": "system", "content": "请帮我翻译一下："},
                {"role": "user", "content": cleaned_text}
            ],
            temperature=0,  # 降低随机性，保证翻译准确性
            max_tokens=500  # 翻译可能需要更长的输出长度
        )

        # 提取翻译结果
        translated_result = response.choices[0].message.content.strip()
        return translated_result

    except Exception as e:
        # 捕获API调用异常
        return f"翻译失败：{str(e)}"


# 测试代码（可直接运行验证）
if __name__ == "__main__":
    translate_test_input = "学而时习之，不亦说乎？"
    translate_test_output = translate_classic_text(translate_test_input)
    print("翻译输入：", translate_test_input)
    print("翻译输出：", translate_test_output)