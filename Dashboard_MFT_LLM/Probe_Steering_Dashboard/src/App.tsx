import React, { useState } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { MoralGauge } from './components/MoralGauge';
import { TimelineChart } from './components/TimelineChart';
import type { ChatMessage, ProbeData } from './types/probe';

// Simulated probe analysis - replace with actual probe implementation
const analyzeWithProbes = (text: string): ProbeData => ({
  harm: Math.random(),
  fairness: Math.random(),
  loyalty: Math.random(),
  authority: Math.random(),
  purity: Math.random(),
  timestamp: Date.now(),
});

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [probeHistory, setProbeHistory] = useState<ProbeData[]>([]);

  const handleSendMessage = (content: string) => {
    // Add user message
    const userProbeData = analyzeWithProbes(content);
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content,
      role: 'user',
      probeData: userProbeData,
    };
    
    // Simulate LLM response
    const assistantResponse = "This is a simulated response. Replace with actual LLM integration.";
    const assistantProbeData = analyzeWithProbes(assistantResponse);
    const assistantMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      content: assistantResponse,
      role: 'assistant',
      probeData: assistantProbeData,
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setProbeHistory(prev => [...prev, userProbeData, assistantProbeData]);
  };

  const currentProbe = probeHistory[probeHistory.length - 1] || {
    harm: 0,
    fairness: 0,
    loyalty: 0,
    authority: 0,
    purity: 0,
    timestamp: Date.now(),
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto grid grid-cols-3 gap-6">
        <div className="col-span-2 h-[calc(100vh-3rem)]">
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
          />
        </div>
        <div className="space-y-6">
          <div className="bg-white p-4 rounded-lg shadow-lg space-y-4">
            <h2 className="text-xl font-bold mb-4">Moral Dimensions</h2>
            <MoralGauge label="Harm" value={currentProbe.harm} color="bg-red-500" />
            <MoralGauge label="Fairness" value={currentProbe.fairness} color="bg-green-500" />
            <MoralGauge label="Loyalty" value={currentProbe.loyalty} color="bg-blue-500" />
            <MoralGauge label="Authority" value={currentProbe.authority} color="bg-yellow-500" />
            <MoralGauge label="Purity" value={currentProbe.purity} color="bg-purple-500" />
          </div>
          <TimelineChart data={probeHistory} />
        </div>
      </div>
    </div>
  );
}

export default App;