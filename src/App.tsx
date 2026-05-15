import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './components/Dashboard';
import { StudentData } from './components/StudentData';
import { AIAssistant } from './components/AIAssistant';
import { EmotionRecognition } from './components/EmotionRecognition';
import { Student } from './types';
import { MOCK_STUDENTS } from './utils/scl90';

export default function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');
  const [students, setStudents] = useState<Student[]>(MOCK_STUDENTS);
  const [selectedStudentForAI, setSelectedStudentForAI] = useState<Student | null>(null);

  const handleSelectStudentForAI = (student: Student) => {
    setSelectedStudentForAI(student);
    setCurrentTab('ai');
  };

  return (
    <div className="flex h-screen bg-sky-50/30 overflow-hidden font-sans text-sky-900">
      <Sidebar currentTab={currentTab} setCurrentTab={setCurrentTab} />
      
      <main className="flex-1 overflow-y-auto relative">
        {currentTab === 'dashboard' && <Dashboard students={students} />}
        {currentTab === 'students' && (
          <StudentData 
            students={students} 
            setStudents={setStudents} 
            onSelectStudentForAI={handleSelectStudentForAI} 
          />
        )}
        {currentTab === 'ai' && <AIAssistant selectedStudent={selectedStudentForAI} />}
        {currentTab === 'emotion' && <EmotionRecognition />}
      </main>
    </div>
  );
}
