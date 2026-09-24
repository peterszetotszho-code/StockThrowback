import type { EChartsOption } from 'echarts';
import type { BacktestResponse } from '../types';
import { EChart } from './EChart';

const COLORS = ['#38bdf8', '#f59e0b', '#a78bfa'];

interface EquityChartProps {
  response: BacktestResponse;
}

export function EquityChart({ response }: EquityChartProps) {
  const names = Object.keys(response.equity_curves);
  if (names.length === 0) return null;

  const option: EChartsOption = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: '#94a3b8' }, top: 0 },
    grid: [
      { left: 60, right: 60, top: 40, height: '45%' },
      { left: 60, right: 60, top: '68%', height: '22%' },
    ],
    xAxis: [
      { type: 'time', gridIndex: 0, axisLabel: { color: '#64748b' } },
      { type: 'time', gridIndex: 1, axisLabel: { color: '#64748b' } },
    ],
    yAxis: [
      {
        gridIndex: 0,
        scale: true,
        axisLabel: { color: '#64748b' },
        splitLine: { lineStyle: { color: '#1e293b' } },
      },
      {
        gridIndex: 1,
        axisLabel: { color: '#64748b', formatter: '{value}%' },
        splitLine: { lineStyle: { color: '#1e293b' } },
      },
    ],
    series: [
      ...names.map((name, i) => ({
        name,
        type: 'line' as const,
        xAxisIndex: 0,
        yAxisIndex: 0,
        showSymbol: false,
        lineStyle: { color: COLORS[i % COLORS.length], width: 1.5 },
        data: response.equity_curves[name].map((p) => [p.time * 1000, p.equity] as [number, number]),
      })),
      ...names.map((name, i) => ({
        name: `${name} DD`,
        type: 'line' as const,
        xAxisIndex: 1,
        yAxisIndex: 1,
        showSymbol: false,
        lineStyle: { color: COLORS[i % COLORS.length], width: 1 },
        areaStyle: { opacity: 0.15 },
        data: response.equity_curves[name].map((p) => [p.time * 1000, p.drawdown] as [number, number]),
      })),
    ],
  };

  return <EChart option={option} />;
}
