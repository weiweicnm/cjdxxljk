import React from 'react';
import { LayoutDashboard, Users, MessageCircleHeart, Leaf, Smile } from 'lucide-react';
import { clsx } from 'clsx';

interface SidebarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
}

export function Sidebar({ currentTab, setCurrentTab }: SidebarProps) {
  const tabs = [
    { id: 'dashboard', name: '数据概览', icon: LayoutDashboard },
    { id: 'students', name: '学生心理档案', icon: Users },
    { id: 'ai', name: '长大 AI 助手', icon: MessageCircleHeart },
    { id: 'emotion', name: '情绪识别', icon: Smile },
  ];

  return (
    <div className="w-64 bg-sky-800 text-sky-50 flex flex-col h-full shadow-xl">
      <div className="p-6 flex items-center gap-3">
        <div className="bg-sky-100 p-2 rounded-xl text-sky-800">
          <Leaf className="w-6 h-6" />
        </div>
        <h1 className="text-xl font-bold tracking-wider">长大心理</h1>
      </div>
      
      <nav className="flex-1 px-4 py-6 space-y-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = currentTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setCurrentTab(tab.id)}
              className={clsx(
                'w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200',
                isActive 
                  ? 'bg-sky-100 text-sky-900 shadow-sm font-medium' 
                  : 'hover:bg-sky-700/50 text-sky-100/80 hover:text-sky-50'
              )}
            >
              <Icon className="w-5 h-5" />
              {tab.name}
            </button>
          );
        })}
      </nav>

      <div className="p-4 text-sm text-sky-400/60 text-center">
        &copy; 2026 长大心理诊疗平台
      </div>
    </div>
  );
}
