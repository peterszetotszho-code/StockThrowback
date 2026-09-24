import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { runReport } from '../api';
import type { ReportResponse } from '../types';

interface ReportPanelProps {
  ticker: string;
  start: string;
  end: string;
}

export function ReportPanel({ ticker, start, end }: ReportPanelProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ReportResponse | null>(null);
  const [newsQuery, setNewsQuery] = useState('');

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await runReport({ ticker, start, end, top_k: 5, news_query: newsQuery.trim() || undefined }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="report-actions">
        <button onClick={handleGenerate} disabled={loading}>
          {loading ? 'Generating…' : 'Generate RAG report'}
        </button>
        <input
          value={newsQuery}
          onChange={(e) => setNewsQuery(e.target.value)}
          placeholder="Company name for news (optional)"
        />
        {data && (
          <div className="badges">
            <span className="badge">
              citations {data.citation.supported}/{data.citation.total_citations}
            </span>
            <span className="badge">coverage {(data.citation.coverage * 100).toFixed(0)}%</span>
            <span className="badge">
              {data.usage.calls} calls · {data.usage.latency_ms.toFixed(0)}ms
            </span>
            <span className="badge">cost ${data.usage.cost_usd.toFixed(6)}</span>
          </div>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      {data && (
        <div className="report-body">
          <div className="report-markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.report}</ReactMarkdown>
          </div>
          <div className="chunk-list">
            <h3>Retrieved chunks ({data.chunks.length})</h3>
            {data.chunks.map((chunk, i) => (
              <div key={chunk.id} className="chunk">
                <div className="chunk-meta">
                  <span className="chunk-index">[{i + 1}]</span>
                  <span>{chunk.source_type}</span>
                  <span>{chunk.date ?? 'n/a'}</span>
                  {chunk.rerank_score != null && (
                    <span>score {chunk.rerank_score.toFixed(2)}</span>
                  )}
                </div>
                <p>{chunk.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
