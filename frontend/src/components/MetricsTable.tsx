import type { MetricRow } from '../types';

function fmt(value: number | null, suffix = '', digits = 2): string {
  if (value == null) return '-';
  return `${value.toFixed(digits)}${suffix}`;
}

interface MetricsTableProps {
  rows: MetricRow[];
}

export function MetricsTable({ rows }: MetricsTableProps) {
  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>Strategy</th>
          <th>Return</th>
          <th>Sharpe</th>
          <th>Max Drawdown</th>
          <th>Win Rate</th>
          <th>Profit Factor</th>
          <th>Trades</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.strategy}>
            <td>{row.strategy}</td>
            <td className={row.return_pct != null && row.return_pct >= 0 ? 'pos' : 'neg'}>
              {fmt(row.return_pct, '%')}
            </td>
            <td>{fmt(row.sharpe)}</td>
            <td className="neg">{fmt(row.max_drawdown_pct, '%')}</td>
            <td>{fmt(row.win_rate_pct, '%')}</td>
            <td>{fmt(row.profit_factor)}</td>
            <td>{row.trades}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
