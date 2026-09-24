import type { BacktestRequest, BacktestResponse, ReportResponse } from './types';

async function post<T>(url: string, payload: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export function runBacktest(payload: BacktestRequest): Promise<BacktestResponse> {
  return post<BacktestResponse>('/api/backtest', payload);
}

export interface ReportRequest {
  ticker: string;
  start: string;
  end: string;
  query?: string;
  top_k?: number;
  news_query?: string;
  max_news?: number;
}

export function runReport(payload: ReportRequest): Promise<ReportResponse> {
  return post<ReportResponse>('/api/report', payload);
}
