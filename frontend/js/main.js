// 导入接口函数
import {
    punctuateText,
    translateText,
    searchPoem,
    getQuizQuestion,
    submitQuizAnswer
} from './api.js';

// DOM元素获取
const navBtns = document.querySelectorAll('.nav-btn');
const modules = document.querySelectorAll('.module');
const loading = document.getElementById('loading');
const toast = document.getElementById('toast');

// 全局变量（存储猜注释的目标词）
let currentQuizWord = "";
let currentCorrectAnswer = ""; // 新增：存储当前题目的正确答案

// ========== 通用工具函数 ==========
// 显示加载动画
const showLoading = () => { loading.style.display = "block"; };
// 隐藏加载动画
const hideLoading = () => { loading.style.display = "none"; };
// 显示提示框
const showToast = (message, type = "error") => {
    toast.textContent = message;
    toast.style.backgroundColor = type === "error" ? "#e53e3e" : "#38a169";
    toast.style.display = "block";
    setTimeout(() => { toast.style.display = "none"; }, 3000);
};
// 复制文本到剪贴板
const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
        showToast("复制成功！", "success");
    }).catch(() => {
        showToast("复制失败，请手动复制");
    });
};

// ========== 模块切换逻辑 ==========
navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // 移除所有激活状态
        navBtns.forEach(b => b.classList.remove('active'));
        modules.forEach(m => m.classList.remove('active'));
        // 添加当前激活状态
        btn.classList.add('active');
        const targetId = btn.dataset.target;
        document.getElementById(targetId).classList.add('active');
        
        // 猜注释模块切换时自动出题
        if (targetId === "quiz") {
            getNewQuizQuestion();
        }
    });
});

// ========== 断句功能逻辑 ==========
const punctuateBtn = document.getElementById('punctuate-btn');
const punctuateInput = document.getElementById('punctuate-input');
const punctuateResult = document.getElementById('punctuate-result');
const punctuateResultContent = punctuateResult.querySelector('.result-content');
const copyBtn = punctuateResult.querySelector('.copy-btn');

// 断句按钮点击事件
punctuateBtn.addEventListener('click', async () => {
    const text = punctuateInput.value.trim();
    if (!text) {
        showToast("请输入需要断句的文本！");
        return;
    }

    showLoading();
    const result = await punctuateText(text);
    hideLoading();

    if (result.success) {
        punctuateResultContent.textContent = result.data.punctuatedText;
        punctuateResult.style.display = "block";
    } else {
        showToast(result.message);
    }
});

// 复制按钮点击事件
copyBtn.addEventListener('click', () => {
    copyToClipboard(punctuateResultContent.textContent);
});

// ========== 翻译功能逻辑 ==========
const translateBtn = document.getElementById('translate-btn');
const translateInput = document.getElementById('translate-input');
const translateResult = document.getElementById('translate-result');
const originalContent = translateResult.querySelector('.original');
const translationContent = translateResult.querySelector('.translation');

// 翻译按钮点击事件
translateBtn.addEventListener('click', async () => {
    const text = translateInput.value.trim();
    if (!text) {
        showToast("请输入需要翻译的文本！");
        return;
    }

    showLoading();
    const result = await translateText(text);
    hideLoading();

    if (result.success) {
        originalContent.textContent = text;
        translationContent.textContent = result.data.translation;
        translateResult.style.display = "block";
    } else {
        showToast(result.message);
    }
});

// ========== 检索功能逻辑 ==========
const searchBtn = document.getElementById('search-btn');
const searchInput = document.getElementById('search-input');
const searchResult = document.getElementById('search-result');
const poemList = searchResult.querySelector('.poem-list');

// 检索按钮点击事件
searchBtn.addEventListener('click', async () => {
    const query = searchInput.value.trim();
    if (!query) {
        showToast("请输入检索关键词！");
        return;
    }

    showLoading();
    const result = await searchPoem(query);
    hideLoading();

    if (result.success) {
        poemList.innerHTML = ""; // 清空原有结果
        const poems = result.data.list;
        if (poems.length === 0) {
            poemList.innerHTML = "<div style='text-align:center; color:#718096;'>未找到相关诗词，请更换关键词重试</div>";
        } else {
            poems.forEach(poem => {
                const poemItem = document.createElement('div');
                poemItem.className = 'poem-item';
                poemItem.innerHTML = `
                    <div class="matched_sentence">${poem.matched_sentence}</div>
                    <div class="title">${poem.name}</div>
                    <div class="dynasty">${poem.dynasty}</div>
                    <div class="author">${poem.author}</div>
                    <div class="content">${poem.content}</div>
                `;
                poemList.appendChild(poemItem);
            });
        }
        searchResult.style.display = "block";
    } else {
        showToast(result.message);
    }
});

// ========== 猜注释功能逻辑 ==========
const quizQuestion = document.getElementById('quiz-question');
const quizInput = document.getElementById('quiz-input');
const quizSubmitBtn = document.getElementById('quiz-submit-btn');
const quizNextBtn = document.getElementById('quiz-next-btn');
const quizResult = document.getElementById('quiz-result');
const scoreEle = quizResult.querySelector('.score');
const correctAnswerEle = quizResult.querySelector('.correct-answer');
// 详细信息元素
const entityScoreEle = quizResult.querySelector('.entity-score');
const penaltyEle = quizResult.querySelector('.penalty');
const errorCountEle = quizResult.querySelector('.error-count');
const feedbackMsgEle = quizResult.querySelector('.feedback-message');
// 实体比对元素
const userEntitiesEle = quizResult.querySelector('.user-entities');
const correctEntitiesEle = quizResult.querySelector('.correct-entities');
// 关系比对元素
const userRelationsEle = quizResult.querySelector('.user-relations');
const correctRelationsEle = quizResult.querySelector('.correct-relations');
const errorRelationsEle = quizResult.querySelector('.error-relations ul');
// 匹配实体对元素
const matchedPairsEle = quizResult.querySelector('.matched-entity-pairs');

// 获取新题目
async function getNewQuizQuestion() {
    showLoading();
    const result = await getQuizQuestion();
    hideLoading();

    if (result.success) {
        const { text, targetWord, correctAnswer } = result.data; // 新增：获取correctAnswer
        currentQuizWord = targetWord;
        currentCorrectAnswer = correctAnswer; // 新增：存储正确答案
        // 高亮目标词（原有逻辑不变）
        const highlightedText = text.replace(targetWord, `<span class="target-word">${targetWord}</span>`);
        quizQuestion.innerHTML = `请说出下列句子中高亮词语的含义：<br><br>${highlightedText}`;
        // 重置输入和结果（原有逻辑不变）
        quizInput.value = "";
        quizResult.style.display = "none";
        quizSubmitBtn.style.display = "block";
        quizNextBtn.style.display = "none";
    } else {
        showToast(result.message);
    }
}

// 提交答案
quizSubmitBtn.addEventListener('click', async () => {
    const userAnswer = quizInput.value.trim();
    if (!userAnswer) {
        showToast("请输入你猜测的释义！");
        return;
    }

    showLoading();
    const result = await submitQuizAnswer(currentQuizWord, userAnswer, currentCorrectAnswer);
    hideLoading();

    if (result.success) {
        const data = result.data;
        // 基础信息展示
        scoreEle.textContent = `你的得分：${data.score}分 (${data.isCorrect ? '正确' : '错误'})`;
        correctAnswerEle.textContent = `正确释义：${data.correctAnswer}`;
        
        // 2. 渲染详细评分信息
        entityScoreEle.textContent = data.entity_score + "分";
        penaltyEle.textContent = data.penalty + "分";
        errorCountEle.textContent = data.error_count + "个";
        feedbackMsgEle.textContent = data.feedback.replace(/\n/g, '<br>'); // 处理换行

        // 3. 渲染实体比对（修复：处理user_entities是数字的情况）
        userEntitiesEle.innerHTML = ""; // 清空原有内容
        if (typeof data.user_entities === 'number' && data.user_entities === 0) {
            userEntitiesEle.innerHTML = "<li>无</li>";
        } else if (Array.isArray(data.user_entities)) {
            data.user_entities.forEach(entity => {
                userEntitiesEle.innerHTML += `<li>${entity}</li>`;
            });
        }
        // 渲染正确实体
        correctEntitiesEle.innerHTML = "";
        data.correct_entities.forEach(entity => {
            correctEntitiesEle.innerHTML += `<li>${entity}</li>`;
        });

        // 4. 渲染关系比对（修复：格式化对象数组）
        userRelationsEle.innerHTML = "";
        data.user_relations.forEach(rel => {
            userRelationsEle.innerHTML += `<li>${rel.head} ${rel.relation} ${rel.tail}（置信度：${rel.confidence}）</li>`;
        });
        // 渲染正确关系
        correctRelationsEle.innerHTML = "";
        data.correct_relations.forEach(rel => {
            correctRelationsEle.innerHTML += `<li>${rel.head} ${rel.relation} ${rel.tail}（置信度：${rel.confidence}）</li>`;
        });

        // 5. 渲染错误关系
        errorRelationsEle.innerHTML = "";
        if (data.error_relations.length === 0) {
            errorRelationsEle.innerHTML = "<li>无</li>";
        } else {
            data.error_relations.forEach(rel => {
                errorRelationsEle.innerHTML += `<li>${rel.head} ${rel.relation} ${rel.tail}</li>`;
            });
        }

        // 6. 渲染匹配实体对（修复：格式化二维数组）
        matchedPairsEle.innerHTML = "";
        data.matched_entity_pairs.forEach(pair => {
            matchedPairsEle.innerHTML += `<li>${pair[0]} ↔ ${pair[1]}（相似度：${pair[2]}）</li>`;
        });
        
        // 显示结果区域并切换按钮
        quizResult.style.display = "block";
        quizSubmitBtn.style.display = "none";
        quizNextBtn.style.display = "block";
    } else {
        showToast(result.message);
    }
});

// 再来一题（原有逻辑不变）
quizNextBtn.addEventListener('click', () => {
    getNewQuizQuestion();
});