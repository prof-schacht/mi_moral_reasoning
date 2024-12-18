export interface ProbeData {
  harm: number;
  fairness: number;
  loyalty: number;
  authority: number;
  purity: number;
  timestamp: number;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  probeData: ProbeData;
}