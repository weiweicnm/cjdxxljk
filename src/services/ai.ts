import { ChatMessage } from '../types';
import OpenAI from 'openai';

// 读取 .env 里的 VITE_GEMINI_API_KEY 存千问key
const apiKey = import.meta.env.VITE_GEMINI_API_KEY;

if (!apiKey) {
  throw new Error('API Key 未配置，请检查 .env');
}

// 千问官方兼容地址 + 开启浏览器模式
const openai = new OpenAI({
  apiKey,
  baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  dangerouslyAllowBrowser: true // 关键配置，解决报错
});

export async function getAIResponse(history: ChatMessage[], context?: string) {
  const systemInstruction = `你是一个校园心理诊疗平台的AI助手，名字叫“长大”。你的性格温馨、治愈、充满生机。你的主要任务是：
1. 为心理老师或辅导员提供关于学生SCL-90测试数据的分析建议。
2. 提供心理疏导的建议和话术。
3. 保持同理心和专业性。
${context ? `当前关注的学生上下文信息：\n${context}` : ''}`;

  const messages = [
    { role: 'system', content: systemInstruction },
    ...history.map(item => ({
      role: item.role === 'assistant' ? 'assistant' : 'user',
      content: item.content
    }))
  ];

  const res = await openai.chat.completions.create({
    model: 'qwen-plus',
    messages
  });

  return res.choices[0]?.message.content || '';
}
