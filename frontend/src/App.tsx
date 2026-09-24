import { useState } from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import { runBacktest } from './api';
import { EquityChart } from './components/EquityChart';
import { MetricsTable } from './components/MetricsTable';
import { PriceChart } from './components/PriceChart';
import { ReportPanel } from './components/ReportPanel';
import type { BacktestResponse } from './types';

const STRATEGIES = ['MA', 'MACD', 'Composite'];

const REGIONS = [
  { label: 'Hong Kong', suffix: 'HK' },
  { label: 'United States', suffix: '' },
  { label: 'Shanghai (China)', suffix: 'SS' },
  { label: 'Shenzhen (China)', suffix: 'SZ' },
  { label: 'Taiwan', suffix: 'TW' },
  { label: 'Japan', suffix: 'T' },
  { label: 'United Kingdom', suffix: 'L' },
];

function toYMD(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function parseYMD(ymd: string): Date {
  return new Date(`${ymd}T00:00:00`);
}

export default function App() {
  const [code, setCode] = useState('0700');
  const [exchange, setExchange] = useState('HK');
  const [start, setStart] = useState('2022-01-01');
  const [end, setEnd] = useState('2024-12-31');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<BacktestResponse | null>(null);

  // Combine code + exchange into a Yahoo ticker (exchange is optional for US stocks).
  const ticker = (code.trim() + (exchange.trim() ? '.' + exchange.trim() : '')).toUpperCase();

  const handleRun = async () => {
    if (!code.trim()) {
      setError('Please enter a stock code.');
      return;
    }
    if (start > end) {
      setError('Start date must be on or before the end date.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await runBacktest({
        ticker,
        start,
        end,
        strategies: STRATEGIES,
        cash: 100000,
        commission: 0.001,
      });
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>stock-trend-lab</h1>
        <div className="controls">
          <input
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder="Code (e.g. 0700)"
            title="Stock code without the exchange suffix"
          />
          <select
            value={exchange}
            onChange={(e) => setExchange(e.target.value)}
            title="Market / exchange"
          >
            {REGIONS.map((region) => (
              <option key={region.suffix || 'US'} value={region.suffix}>
                {region.label}
              </option>
            ))}
          </select>
          <DatePicker
            selected={parseYMD(start)}
            onChange={(date) => date && setStart(toYMD(date))}
            dateFormat="yyyy-MM-dd"
            className="date-input"
            showMonthDropdown
            showYearDropdown
            scrollableYearDropdown
            yearDropdownItemNumber={30}
            maxDate={new Date()}
          />
          <DatePicker
            selected={parseYMD(end)}
            onChange={(date) => date && setEnd(toYMD(date))}
            dateFormat="yyyy-MM-dd"
            className="date-input"
            showMonthDropdown
            showYearDropdown
            scrollableYearDropdown
            yearDropdownItemNumber={30}
            maxDate={new Date()}
          />
          <button onClick={handleRun} disabled={loading}>
            {loading ? 'Running…' : 'Run backtest'}
          </button>
        </div>
      </header>

      <details className="glossary">
        <summary>Indicators &amp; strategies explained</summary>
        <div className="glossary-body">
          <dl>
            <dt>MA (Moving Average)</dt>
            <dd>
              Average close price over N days (MA5 = last 5 days). A short MA crossing
              above a long MA is a &quot;golden cross&quot; (buy); the reverse is a
              &quot;death cross&quot; (sell).
            </dd>
            <dt>MACD</dt>
            <dd>
              Momentum = fast EMA (12) minus slow EMA (26), plus a 9-day signal line.
              Positive MACD means bullish momentum.
            </dd>
            <dt>RSI</dt>
            <dd>
              Relative Strength Index (0–100). Above 70 = overbought, below 30 = oversold.
            </dd>
            <dt>Bollinger Bands</dt>
            <dd>
              A middle moving average plus/minus 2 standard deviations. Price near the
              upper band = potentially overbought.
            </dd>
            <dt>Trend state</dt>
            <dd>
              bullish (MA5 &gt; MA20 and MACD &gt; 0), bearish (the opposite), or ranging
              (otherwise).
            </dd>
            <dt>Strategies</dt>
            <dd>
              MA = golden cross buy / death cross sell. MACD = MACD crosses its signal
              line. Composite = golden cross plus RSI below 70.
            </dd>
          </dl>
        </div>
      </details>

      {error && <div className="error">{error}</div>}

      {data && (
        <main>
          <section>
            <h2>
              Price &amp; indicators — {data.ticker}
              {data.company_name ? ` · ${data.company_name}` : ''}
              {data.currency ? ` · ${data.currency}` : ''}
            </h2>
            <PriceChart candles={data.candles} />
          </section>
          <section>
            <h2>Equity &amp; drawdown</h2>
            <EquityChart response={data} />
          </section>
          <section>
            <h2>Strategy comparison</h2>
            <MetricsTable rows={data.comparison} />
          </section>
          <section>
            <h2>Failure analysis</h2>
            {Object.entries(data.failures).map(([name, text]) => (
              <div key={name} className="failure">
                <strong>{name}</strong>
                <p>{text}</p>
              </div>
            ))}
          </section>
        </main>
      )}

      <section>
        <h2>RAG report</h2>
        <ReportPanel ticker={ticker} start={start} end={end} />
      </section>
    </div>
  );
}
