import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './App.css';

interface PredictionResponse {
  loan_id: string;
  next_3m_delinquency_prob: number;
  next_12m_default_prob: number;
  next_12m_prepayment_prob: number;
  is_anomaly: boolean;
  reviewer_summary: string;
}

function App() {
  const [loanId, setLoanId] = useState('100010079393');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ loan_id: loanId, features: {} })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error predicting", error);
    }
    setLoading(false);
  };

  const getChartData = () => {
    if (!result) return [];
    return [
      { name: '3M Delinquency', value: result.next_3m_delinquency_prob * 100, color: '#f59e0b' },
      { name: '12M Default', value: result.next_12m_default_prob * 100, color: '#ef4444' },
      { name: '12M Prepayment', value: result.next_12m_prepayment_prob * 100, color: '#3b82f6' },
    ];
  };

  return (
    <div style={{ backgroundColor: '#f8fafc', minHeight: '100vh', padding: '3rem 2rem', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: '800', color: '#0f172a', marginBottom: '0.5rem' }}>
            Loan Performance Intelligence Engine 🚀
          </h1>
          <p style={{ fontSize: '1.1rem', color: '#64748b' }}>AI-Powered Portfolio Risk & Reviewer Copilot</p>
        </div>
        
        {/* Input Form */}
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '3rem' }}>
          <div style={{ display: 'flex', gap: '1rem', background: 'white', padding: '1rem', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)' }}>
            <input 
              value={loanId}
              onChange={(e) => setLoanId(e.target.value)}
              placeholder="Enter 12-Digit Loan ID"
              style={{ padding: '0.75rem 1rem', fontSize: '1.1rem', border: '1px solid #cbd5e1', borderRadius: '8px', outline: 'none', width: '300px' }}
            />
            <button 
              onClick={handlePredict} 
              disabled={loading}
              style={{ padding: '0.75rem 2rem', fontSize: '1.1rem', fontWeight: '600', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '8px', cursor: loading ? 'not-allowed' : 'pointer', transition: 'background 0.2s', opacity: loading ? 0.7 : 1 }}
            >
              {loading ? 'Analyzing...' : 'Analyze Risk'}
            </button>
          </div>
        </div>

        {/* Results Grid */}
        {result && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '2rem' }}>
            
            {/* Risk Chart Card */}
            <div style={{ background: 'white', padding: '2rem', borderRadius: '16px', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <h3 style={{ margin: 0, fontSize: '1.25rem', color: '#1e293b' }}>Risk Probabilities</h3>
                <span style={{ padding: '0.5rem 1rem', borderRadius: '9999px', fontSize: '0.875rem', fontWeight: '600', background: result.is_anomaly ? '#fee2e2' : '#dcfce7', color: result.is_anomaly ? '#991b1b' : '#166534' }}>
                  {result.is_anomaly ? '⚠️ Anomaly Detected' : '✅ Standard Profile'}
                </span>
              </div>
              <div style={{ height: '300px', width: '100%' }}>
                <ResponsiveContainer>
                  <BarChart data={getChartData()} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <XAxis dataKey="name" axisLine={false} tickLine={false} />
                    <YAxis axisLine={false} tickLine={false} domain={[0, 100]} tickFormatter={(val) => `${val}%`} />
                    <Tooltip cursor={{ fill: 'transparent' }} formatter={(value: number) => [`${value.toFixed(1)}%`, 'Probability']} />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {getChartData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* LLM Copilot Card */}
            <div style={{ background: 'linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%)', padding: '2rem', borderRadius: '16px', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)', display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                <span style={{ fontSize: '1.5rem' }}>🤖</span>
                <h3 style={{ margin: 0, fontSize: '1.25rem', color: '#1e3a8a' }}>LLM Reviewer Copilot</h3>
              </div>
              <div style={{ background: 'white', padding: '1.5rem', borderRadius: '12px', flexGrow: 1, border: '1px solid #bfdbfe' }}>
                <p style={{ whiteSpace: 'pre-wrap', lineHeight: '1.7', color: '#334155', margin: 0, fontSize: '1.05rem' }}>
                  {result.reviewer_summary}
                </p>
              </div>
              <div style={{ marginTop: '1.5rem', fontSize: '0.875rem', color: '#64748b', textAlign: 'center' }}>
                ✨ Summary generated via RAG against data_dictionary.md
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}

export default App;
