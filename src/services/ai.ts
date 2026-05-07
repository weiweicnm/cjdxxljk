import { GoogleGenAI } from '@google/genai';
import { ChatMessage } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

export async function getAIResponse(history: ChatMessage[], context?: string) {
  const systemInstruction = `你是一个校园心理诊疗平台的AI助手，名字叫“长大”。你的性格温馨、治愈、充满生机。
你的主要任务是：
1. 为心理老师或辅导员提供关于学生SCL-90测试数据的分析建议。
2. 提供心理疏导的建议和话术。
3. 保持同理心和专业性。
${context ? `当前关注的学生上下文信息：\n${context}` : ''}`;

  const contents = history.map(msg => ({
    role: msg.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: msg.content }]
  }));

  const response = await ai.models.generateContent({
    model: 'gemini-3.1-pro-preview',
    contents: contents,
    config: {
      systemInstruction,
    }
  });

  return response.text;
}
