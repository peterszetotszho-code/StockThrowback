import type { BacktestRequest, BacktestResponse, ReportResponse } from './types';

async function post<T>(url: string, payload: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(await readError(res));
  }
  return res.json();
}

async function readError(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const data = JSON.parse(text);
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join('; ');
    }
  } catch {
    // not JSON; fall through
  }
  return text || `API ${res.status}`;
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
