import React, { useState } from 'react';
import { Student } from '../types';
import { Upload, Search, FileText, AlertTriangle, CheckCircle2 } from 'lucide-react';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { calculateWarningLevel } from '../utils/scl90';

interface StudentDataProps {
  students: Student[];
  setStudents: React.Dispatch<React.SetStateAction<Student[]>>;
  onSelectStudentForAI: (student: Student) => void;
}

export function StudentData({ students, setStudents, onSelectStudentForAI }: StudentDataProps) {
  const [searchTerm, setSearchTerm] = useState('');

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const processData = (data: any[]) => {
      const newStudents: Student[] = data.map((row: any, index) => {
        const scl90 = {
          somatization: parseFloat(row['SCL90-躯体化'] || '0'),
          obsessiveCompulsive: parseFloat(row['SCL90-强迫症状'] || '0'),
          interpersonalSensitivity: parseFloat(row['SCL90-人际关系敏感'] || '0'),
          depression: parseFloat(row['SCL90-抑郁'] || '0'),
          anxiety: parseFloat(row['SCL90-焦虑'] || '0'),
          hostility: parseFloat(row['SCL90-敌对'] || '0'),
          phobicAnxiety: parseFloat(row['SCL90-恐怖'] || '0'),
          paranoidIdeation: parseFloat(row['SCL90-偏执'] || '0'),
          psychoticism: parseFloat(row['SCL90-精神病性'] || '0'),
          others: parseFloat(row['SCL90-其他'] || '0'),
          totalScore: parseFloat(row['SCL90-总分'] || '0'),
          globalSeverityIndex: parseFloat(row['SCL90-总均分'] || '0'),
          positiveSymptomTotal: parseInt(row['SCL90-阳性项目数'] || '0', 10),
        };

        return {
          id: `imported-${Date.now()}-${index}`,
          name: row['姓名'] || `学生${index + 1}`,
          gender: row['性别'] || '未知',
          ethnicity: row['民族'] || '未知',
          department: row['系'] || '未知',
          major: row['专业'] || '未知',
          origin: row['生源地'] || '未知',
          familyStatus: row['家庭情况'] || '未知',
          familyEconomy: row['家庭经济状况'] || '未知',
          scl90,
          warningLevel: calculateWarningLevel(scl90)
        };
      });

      setStudents(prev => [...prev, ...newStudents]);
    };

    if (file.name.endsWith('.csv')) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          processData(results.data);
        }
      });
    } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        const bstr = evt.target?.result;
        const wb = XLSX.read(bstr, { type: 'binary' });
        const wsname = wb.SheetNames[0];
        const ws = wb.Sheets[wsname];
        const data = XLSX.utils.sheet_to_json(ws);
        processData(data);
      };
      reader.readAsBinaryString(file);
    }
  };

  const filteredStudents = students.filter(s => 
    s.name.includes(searchTerm) || 
    s.department.includes(searchTerm) || 
    s.major.includes(searchTerm) ||
    s.warningLevel.includes(searchTerm)
  );

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-bold text-sky-900">学生心理档案</h2>
          <p className="text-sky-600/80 mt-2">管理和分析学生 SCL-90 测评数据</p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-sky-400 w-5 h-5" />
            <input 
              type="text" 
              placeholder="搜索姓名、系别、预警级别..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2.5 bg-white border border-sky-100 rounded-full focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 w-64 shadow-sm"
            />
          </div>
          
          <label className="cursor-pointer bg-sky-600 hover:bg-sky-700 text-white px-5 py-2.5 rounded-full flex items-center gap-2 transition-colors shadow-sm font-medium">
            <Upload className="w-5 h-5" />
            导入数据 (CSV/Excel)
            <input type="file" accept=".csv,.xlsx,.xls" className="hidden" onChange={handleFileUpload} />
          </label>
        </div>
      </header>

      <div className="bg-white rounded-3xl shadow-sm border border-sky-50 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-sky-50/50 text-sky-800 text-sm uppercase tracking-wider">
                <th className="p-4 font-semibold rounded-tl-3xl">姓名</th>
                <th className="p-4 font-semibold">系别 / 专业</th>
                <th className="p-4 font-semibold">SCL-90 总分</th>
                <th className="p-4 font-semibold">预警级别</th>
                <th className="p-4 font-semibold rounded-tr-3xl text-right">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-sky-50">
              {filteredStudents.map((student) => (
                <tr key={student.id} className="hover:bg-sky-50/30 transition-colors">
                  <td className="p-4">
                    <div className="font-medium text-sky-900">{student.name}</div>
                    <div className="text-sm text-sky-600/70">{student.gender} &middot; {student.ethnicity}</div>
                  </td>
                  <td className="p-4">
                    <div className="text-sky-800">{student.department}</div>
                    <div className="text-sm text-sky-600/70">{student.major}</div>
                  </td>
                  <td className="p-4">
                    <div className="font-mono text-sky-900">{student.scl90.totalScore}</div>
                    <div className="text-sm text-sky-600/70">均分: {student.scl90.globalSeverityIndex}</div>
                  </td>
                  <td className="p-4">
                    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium
                      ${student.warningLevel === '正常' ? 'bg-teal-50 text-teal-700 border border-teal-200/50' : ''}
                      ${student.warningLevel === '轻度预警' ? 'bg-amber-50 text-amber-700 border border-amber-200/50' : ''}
                      ${student.warningLevel === '重度预警' ? 'bg-rose-50 text-rose-700 border border-rose-200/50' : ''}
                    `}>
                      {student.warningLevel === '正常' && <CheckCircle2 className="w-4 h-4" />}
                      {student.warningLevel !== '正常' && <AlertTriangle className="w-4 h-4" />}
                      {student.warningLevel}
                    </span>
                  </td>
                  <td className="p-4 text-right">
                    <button 
                      onClick={() => onSelectStudentForAI(student)}
                      className="inline-flex items-center gap-2 px-4 py-2 bg-sky-50 hover:bg-sky-100 text-sky-700 rounded-full text-sm font-medium transition-colors"
                    >
                      <FileText className="w-4 h-4" />
                      AI 咨询建议
                    </button>
                  </td>
                </tr>
              ))}
              {filteredStudents.length === 0 && (
                <tr>
                  <td colSpan={5} className="p-12 text-center text-sky-600/60">
                    暂无学生数据，请导入 CSV/Excel 文件或调整搜索条件。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
