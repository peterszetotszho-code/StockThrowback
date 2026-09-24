import { useEffect, useRef } from 'react';
import { ColorType, UTCTimestamp, createChart } from 'lightweight-charts';
import type { Candle } from '../types';

interface PriceChartProps {
  candles: Candle[];
}

export function PriceChart({ candles }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      height: 420,
      layout: {
        background: { type: ColorType.Solid, color: '#0b1220' },
        textColor: '#94a3b8',
      },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      timeScale: { borderColor: '#334155', timeVisible: true },
      rightPriceScale: { borderColor: '#334155' },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      borderVisible: false,
    });

    const ma5 = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1, priceLineVisible: false });
    const ma20 = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, priceLineVisible: false });
    const ma60 = chart.addLineSeries({ color: '#a78bfa', lineWidth: 1, priceLineVisible: false });

    candleSeries.setData(
      candles.map((c) => ({
        time: c.time as UTCTimestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );

    const lineData = (key: 'ma5' | 'ma20' | 'ma60') =>
      candles
        .filter((c) => c[key] != null)
        .map((c) => ({ time: c.time as UTCTimestamp, value: c[key] as number }));

    ma5.setData(lineData('ma5'));
    ma20.setData(lineData('ma20'));
    ma60.setData(lineData('ma60'));

    const handleResize = () => chart.applyOptions({ width: container.clientWidth });
    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [candles]);

  return <div ref={containerRef} className="chart" />;
}
