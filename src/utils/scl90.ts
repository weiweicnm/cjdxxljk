import { SCL90Data } from '../types';

export function calculateWarningLevel(scl90: SCL90Data): '正常' | '轻度预警' | '重度预警' {
  const factors = [
    scl90.somatization,
    scl90.obsessiveCompulsive,
    scl90.interpersonalSensitivity,
    scl90.depression,
    scl90.anxiety,
    scl90.hostility,
    scl90.phobicAnxiety,
    scl90.paranoidIdeation,
    scl90.psychoticism
  ];

  const anyFactorSevere = factors.some(f => f >= 3);
  const anyFactorMild = factors.some(f => f >= 2);

  if (scl90.totalScore >= 200 || anyFactorSevere) {
    return '重度预警';
  } else if (scl90.totalScore >= 160 || anyFactorMild || scl90.positiveSymptomTotal > 43) {
    return '轻度预警';
  }
  return '正常';
}

export const MOCK_STUDENTS = [
  {
    id: '2023001',
    name: '张三',
    gender: '男',
    ethnicity: '汉族',
    department: '计算机系',
    major: '软件工程',
    origin: '北京',
    familyStatus: '双亲',
    familyEconomy: '中等',
    scl90: {
      somatization: 1.5,
      obsessiveCompulsive: 1.8,
      interpersonalSensitivity: 2.1,
      depression: 1.6,
      anxiety: 1.4,
      hostility: 1.2,
      phobicAnxiety: 1.1,
      paranoidIdeation: 1.3,
      psychoticism: 1.2,
      others: 1.0,
      totalScore: 142,
      globalSeverityIndex: 1.58,
      positiveSymptomTotal: 35
    },
    warningLevel: '轻度预警' as const
  },
  {
    id: '2023002',
    name: '李四',
    gender: '女',
    ethnicity: '回族',
    department: '外语系',
    major: '英语',
    origin: '宁夏',
    familyStatus: '单亲',
    familyEconomy: '困难',
    scl90: {
      somatization: 2.5,
      obsessiveCompulsive: 2.8,
      interpersonalSensitivity: 3.1,
      depression: 3.5,
      anxiety: 3.2,
      hostility: 2.2,
      phobicAnxiety: 2.1,
      paranoidIdeation: 2.3,
      psychoticism: 1.8,
      others: 2.0,
      totalScore: 255,
      globalSeverityIndex: 2.83,
      positiveSymptomTotal: 65
    },
    warningLevel: '重度预警' as const
  },
  {
    id: '2023003',
    name: '王五',
    gender: '男',
    ethnicity: '汉族',
    department: '机械系',
    major: '机械工程',
    origin: '山东',
    familyStatus: '双亲',
    familyEconomy: '良好',
    scl90: {
      somatization: 1.1,
      obsessiveCompulsive: 1.2,
      interpersonalSensitivity: 1.1,
      depression: 1.0,
      anxiety: 1.1,
      hostility: 1.0,
      phobicAnxiety: 1.0,
      paranoidIdeation: 1.0,
      psychoticism: 1.0,
      others: 1.0,
      totalScore: 95,
      globalSeverityIndex: 1.06,
      positiveSymptomTotal: 5
    },
    warningLevel: '正常' as const
  }
];
