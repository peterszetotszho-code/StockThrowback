export interface Candle {
  time: number; // unix seconds (UTC)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ma5: number | null;
  ma20: number | null;
  ma60: number | null;
  rsi: number | null;
  trend: string;
}

export interface EquityPoint {
  time: number;
  equity: number;
  drawdown: number; // negative percent
}

export interface MetricRow {
  strategy: string;
  return_pct: number | null;
  sharpe: number | null;
  max_drawdown_pct: number | null;
  win_rate_pct: number | null;
  profit_factor: number | null;
  trades: number;
}

export interface BacktestResponse {
  ticker: string;
  start: string;
  end: string;
  candles: Candle[];
  comparison: MetricRow[];
  failures: Record<string, string>;
  equity_curves: Record<string, EquityPoint[]>;
}

export interface BacktestRequest {
  ticker: string;
  start: string;
  end: string;
  strategies: string[];
  cash: number;
  commission: number;
}
