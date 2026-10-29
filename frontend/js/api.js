// 接口基础地址（请替换为你的后端实际地址）
const BASE_URL = "http://localhost:8080/api/classic";

// 接口封装 - 断句功能
export const punctuateText = async (text) => {
    try {
        const response = await axios.post(`${BASE_URL}/punctuate`, { text });
        return { success: true, data: response.data };
    } catch (error) {
        return { success: false, message: error.response?.data?.message || "断句请求失败，请重试" };
    }
};

// 接口封装 - 翻译功能
export const translateText = async (text) => {
    try {
        const response = await axios.post(`${BASE_URL}/translate`, { text });
        return { success: true, data: response.data };
    } catch (error) {
        return { success: false, message: error.response?.data?.message || "翻译请求失败，请重试" };
    }
};

// 接口封装 - 检索功能
export const searchPoem = async (query) => {
    try {
        const response = await axios.post(`${BASE_URL}/search`, { query });
        return { success: true, data: response.data };
    } catch (error) {
        return { success: false, message: error.response?.data?.message || "检索请求失败，请重试" };
    }
};

// 接口封装 - 猜注释-出题
export const getQuizQuestion = async () => {
    try {
        const response = await axios.get(`${BASE_URL}/quiz/get`);
        return { success: true, data: response.data };
    } catch (error) {
        return { success: false, message: error.response?.data?.message || "获取题目失败，请重试" };
    }
};

// 接口封装 - 猜注释-提交答案
export const submitQuizAnswer = async (targetWord, userAnswer, correctAnswer) => { // 新增correctAnswer参数
    try {
        // 新增：将correctAnswer加入请求体
        const response = await axios.post(`${BASE_URL}/quiz/submit`, { targetWord, userAnswer, correctAnswer });
        return { success: true, data: response.data };
    } catch (error) {
        return { success: false, message: error.response?.data?.message || "提交答案失败，请重试" };
    }
};