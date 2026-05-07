export interface SCL90Data {
  somatization: number; // 躯体化
  obsessiveCompulsive: number; // 强迫症状
  interpersonalSensitivity: number; // 人际关系敏感
  depression: number; // 抑郁
  anxiety: number; // 焦虑
  hostility: number; // 敌对
  phobicAnxiety: number; // 恐怖
  paranoidIdeation: number; // 偏执
  psychoticism: number; // 精神病性
  others: number; // 其他
  totalScore: number; // 总分
  globalSeverityIndex: number; // 总均分
  positiveSymptomTotal: number; // 阳性项目数
}

export interface Student {
  id: string;
  name: string;
  gender: string;
  ethnicity: string;
  department: string;
  major: string;
  origin: string;
  familyStatus: string;
  familyEconomy: string;
  scl90: SCL90Data;
  warningLevel: '正常' | '轻度预警' | '重度预警';
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
