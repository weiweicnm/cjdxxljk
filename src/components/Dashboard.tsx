import React from 'react';
import { Student } from '../types';
import { ShieldAlert, ShieldCheck, Shield } from 'lucide-react';

interface DashboardProps {
  students: Student[];
}

export function Dashboard({ students }: DashboardProps) {
  const total = students.length;
  const normal = students.filter(s => s.warningLevel === '正常').length;
  const mild = students.filter(s => s.warningLevel === '轻度预警').length;
  const severe = students.filter(s => s.warningLevel === '重度预警').length;

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <header className="mb-8">
        <h2 className="text-3xl font-bold text-sky-900">数据概览</h2>
        <p className="text-sky-600/80 mt-2">全校学生心理健康状态实时监控</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <StatCard 
          title="总测评人数" 
          value={total} 
          icon={<Shield className="w-8 h-8 text-sky-600" />} 
          bg="bg-sky-50"
          border="border-sky-100"
        />
        <StatCard 
          title="状态正常" 
          value={normal} 
          icon={<ShieldCheck className="w-8 h-8 text-teal-500" />} 
          bg="bg-teal-50"
          border="border-teal-100"
        />
        <StatCard 
          title="轻度预警" 
          value={mild} 
          icon={<ShieldAlert className="w-8 h-8 text-amber-500" />} 
          bg="bg-amber-50"
          border="border-amber-100"
        />
        <StatCard 
          title="重度预警" 
          value={severe} 
          icon={<ShieldAlert className="w-8 h-8 text-rose-500" />} 
          bg="bg-rose-50"
          border="border-rose-100"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-12">
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-sky-50">
          <h3 className="text-xl font-semibold text-sky-900 mb-6">预警分布</h3>
          <div className="space-y-4">
            <ProgressBar label="正常" value={normal} total={total} color="bg-teal-400" />
            <ProgressBar label="轻度预警" value={mild} total={total} color="bg-amber-400" />
            <ProgressBar label="重度预警" value={severe} total={total} color="bg-rose-400" />
          </div>
        </div>

        <div className="bg-white p-6 rounded-3xl shadow-sm border border-sky-50">
          <h3 className="text-xl font-semibold text-sky-900 mb-6">近期高频心理因子</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-sky-50/50 rounded-2xl">
              <span className="font-medium text-sky-800">抑郁 (Depression)</span>
              <span className="text-sm text-sky-600 bg-sky-100 px-3 py-1 rounded-full">关注度高</span>
            </div>
            <div className="flex items-center justify-between p-4 bg-sky-50/50 rounded-2xl">
              <span className="font-medium text-sky-800">人际关系敏感 (Interpersonal Sensitivity)</span>
              <span className="text-sm text-sky-600 bg-sky-100 px-3 py-1 rounded-full">关注度中</span>
            </div>
            <div className="flex items-center justify-between p-4 bg-sky-50/50 rounded-2xl">
              <span className="font-medium text-sky-800">焦虑 (Anxiety)</span>
              <span className="text-sm text-sky-600 bg-sky-100 px-3 py-1 rounded-full">关注度中</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ title, value, icon, bg, border }: { title: string, value: number, icon: React.ReactNode, bg: string, border: string }) {
  return (
    <div className={`p-6 rounded-3xl ${bg} border ${border} flex flex-col justify-between h-40 transition-transform hover:-translate-y-1`}>
      <div className="flex justify-between items-start">
        <h3 className="text-gray-600 font-medium">{title}</h3>
        {icon}
      </div>
      <div className="text-4xl font-bold text-gray-900">{value}</div>
    </div>
  );
}

function ProgressBar({ label, value, total, color }: { label: string, value: number, total: number, color: string }) {
  const percentage = total === 0 ? 0 : Math.round((value / total) * 100);
  return (
    <div>
      <div className="flex justify-between text-sm mb-2">
        <span className="text-gray-600 font-medium">{label}</span>
        <span className="text-gray-900 font-bold">{percentage}% ({value}人)</span>
      </div>
      <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all duration-1000`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}
