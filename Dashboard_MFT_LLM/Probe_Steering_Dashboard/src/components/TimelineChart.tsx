import React from 'react';
import type { ProbeData } from '../types/probe';

interface TimelineChartProps {
  data: ProbeData[];
}

export function TimelineChart({ data }: TimelineChartProps) {
  const maxPoints = 50;
  const displayData = data.slice(-maxPoints);

  const chartHeight = 200;
  const chartWidth = '100%';

  return (
    <div className="bg-white p-4 rounded-lg shadow-lg">
      <h3 className="text-lg font-semibold mb-4">Moral Dimensions Timeline</h3>
      <div style={{ height: chartHeight, width: chartWidth }} className="relative">
        {/* Render SVG chart here - simplified for example */}
        <div className="absolute inset-0 flex items-end">
          {displayData.map((point, index) => (
            <div
              key={index}
              className="flex-1 flex flex-col justify-end gap-0.5"
              style={{ height: '100%' }}
            >
              <div
                className="w-full bg-red-500 transition-all duration-300"
                style={{ height: `${point.harm * 100}%` }}
              />
              <div
                className="w-full bg-green-500 transition-all duration-300"
                style={{ height: `${point.fairness * 100}%` }}
              />
              <div
                className="w-full bg-blue-500 transition-all duration-300"
                style={{ height: `${point.loyalty * 100}%` }}
              />
              <div
                className="w-full bg-yellow-500 transition-all duration-300"
                style={{ height: `${point.authority * 100}%` }}
              />
              <div
                className="w-full bg-purple-500 transition-all duration-300"
                style={{ height: `${point.purity * 100}%` }}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}