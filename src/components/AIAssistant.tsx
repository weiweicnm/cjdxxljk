import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, Student } from '../types';
import { getAIResponse } from '../services/ai';
import { Send, Bot, User, Loader2, Sparkles } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { clsx } from 'clsx';

interface AIAssistantProps {
  selectedStudent: Student | null;
}

export function AIAssistant({ selectedStudent }: AIAssistantProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([{
    id: 'welcome',
    role: 'assistant',
    content: '你好，我是长大心理助手。你可以向我询问关于学生心理数据的分析建议，或者探讨心理疏导的话术。',
    timestamp: new Date()
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedStudent) {
      const contextMsg = `我想咨询关于学生 ${selectedStudent.name} 的情况。
基本信息：${selectedStudent.gender}，${selectedStudent.ethnicity}，${selectedStudent.department} ${selectedStudent.major}。
家庭情况：${selectedStudent.familyStatus}，经济状况：${selectedStudent.familyEconomy}。
SCL-90测试结果：总分 ${selectedStudent.scl90.totalScore}，总均分 ${selectedStudent.scl90.globalSeverityIndex}，阳性项目数 ${selectedStudent.scl90.positiveSymptomTotal}。
各因子得分：躯体化 ${selectedStudent.scl90.somatization}，强迫症状 ${selectedStudent.scl90.obsessiveCompulsive}，人际关系敏感 ${selectedStudent.scl90.interpersonalSensitivity}，抑郁 ${selectedStudent.scl90.depression}，焦虑 ${selectedStudent.scl90.anxiety}，敌对 ${selectedStudent.scl90.hostility}，恐怖 ${selectedStudent.scl90.phobicAnxiety}，偏执 ${selectedStudent.scl90.paranoidIdeation}，精神病性 ${selectedStudent.scl90.psychoticism}。
预警级别：${selectedStudent.warningLevel}。
请帮我分析一下该学生的心理状况，并提供一些后续跟进和沟通的建议。`;

      setInput(contextMsg);
    }
  }, [selectedStudent]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const history = [...messages, userMsg];
      const responseText = await getAIResponse(history);
      
      const aiMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: responseText,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error('AI Response Error:', error);
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '抱歉，我遇到了一些网络问题，请稍后再试。',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-2rem)] max-w-5xl mx-auto p-4 md:p-8 animate-in fade-in duration-500">
      <header className="mb-6 flex items-center gap-4">
        <div className="bg-sky-100 p-3 rounded-2xl text-sky-600">
          <Sparkles className="w-8 h-8" />
        </div>
        <div>
          <h2 className="text-3xl font-bold text-sky-900">长大 AI 助手</h2>
          <p className="text-sky-600/80 mt-1">您的专属校园心理咨询辅助专家</p>
        </div>
      </header>

      <div className="flex-1 bg-white rounded-3xl shadow-sm border border-sky-50 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg) => (
            <div 
              key={msg.id} 
              className={clsx(
                "flex gap-4 max-w-[85%]",
                msg.role === 'user' ? "ml-auto flex-row-reverse" : ""
              )}
            >
              <div className={clsx(
                "w-10 h-10 rounded-full flex items-center justify-center shrink-0 shadow-sm",
                msg.role === 'user' ? "bg-sky-600 text-white" : "bg-sky-50 text-sky-600 border border-sky-100"
              )}>
                {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
              </div>
              
              <div className={clsx(
                "p-4 rounded-2xl",
                msg.role === 'user' 
                  ? "bg-sky-600 text-white rounded-tr-sm" 
                  : "bg-sky-50/50 text-sky-900 border border-sky-100/50 rounded-tl-sm"
              )}>
                {msg.role === 'user' ? (
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                ) : (
                  <div className="markdown-body text-sm">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                )}
                <div className={clsx(
                  "text-xs mt-2 opacity-60",
                  msg.role === 'user' ? "text-sky-100 text-right" : "text-sky-600"
                )}>
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-4 max-w-[85%]">
              <div className="w-10 h-10 rounded-full bg-sky-50 text-sky-600 border border-sky-100 flex items-center justify-center shrink-0 shadow-sm">
                <Bot className="w-5 h-5" />
              </div>
              <div className="p-4 rounded-2xl bg-sky-50/50 text-sky-900 border border-sky-100/50 rounded-tl-sm flex items-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin text-sky-600" />
                <span className="text-sky-600/80 text-sm">长大正在思考...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-sky-50/30 border-t border-sky-50">
          <div className="relative flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="输入您的问题或粘贴学生数据..."
              className="w-full bg-white border border-sky-100 rounded-2xl pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 resize-none shadow-sm min-h-[56px] max-h-48"
              rows={input.split('\n').length > 1 ? Math.min(input.split('\n').length, 5) : 1}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="absolute right-2 bottom-2 p-2 bg-sky-600 hover:bg-sky-700 disabled:bg-sky-300 text-white rounded-xl transition-colors shadow-sm"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <div className="text-center mt-2 text-xs text-sky-600/60">
            按 Enter 发送，Shift + Enter 换行
          </div>
        </div>
      </div>
    </div>
  );
}
