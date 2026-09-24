import { useState } from 'react';
import { runBacktest } from './api';
import { EquityChart } from './components/EquityChart';
import { MetricsTable } from './components/MetricsTable';
import { PriceChart } from './components/PriceChart';
import { ReportPanel } from './components/ReportPanel';
import type { BacktestResponse } from './types';

const STRATEGIES = ['MA', 'MACD', 'Composite'];

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
          <input
            value={exchange}
            onChange={(e) => setExchange(e.target.value)}
            placeholder="Exchange (e.g. HK)"
            title="Exchange suffix; leave empty for US stocks (e.g. AAPL)"
          />
          <input value={start} onChange={(e) => setStart(e.target.value)} placeholder="Start" />
          <input value={end} onChange={(e) => setEnd(e.target.value)} placeholder="End" />
          <button onClick={handleRun} disabled={loading}>
            {loading ? 'Running…' : 'Run backtest'}
          </button>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      {data && (
        <main>
          <section>
            <h2>
              Price &amp; indicators — {data.ticker}
              {data.company_name ? ` · ${data.company_name}` : ''}
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
