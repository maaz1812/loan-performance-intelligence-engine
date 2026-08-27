import React, { useState } from 'react';
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

  return (
    <div className="App" style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>Loan Performance Intelligence Engine</h1>
      
      <div style={{ marginBottom: '2rem', padding: '1rem', background: '#f5f5f5', borderRadius: '8px' }}>
        <h2>Loan Assessor</h2>
        <input 
          value={loanId}
          onChange={(e) => setLoanId(e.target.value)}
          placeholder="Enter Loan ID"
          style={{ padding: '0.5rem', marginRight: '1rem', fontSize: '1rem' }}
        />
        <button 
          onClick={handlePredict} 
          disabled={loading}
          style={{ padding: '0.5rem 1rem', fontSize: '1rem', background: '#0070f3', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
        >
          {loading ? 'Analyzing...' : 'Analyze Loan'}
        </button>
      </div>

      {result && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <div style={{ border: '1px solid #ddd', padding: '1.5rem', borderRadius: '8px' }}>
            <h3>Prediction Outputs</h3>
            <p><strong>3M Delinquency:</strong> {(result.next_3m_delinquency_prob * 100).toFixed(1)}%</p>
            <p><strong>12M Default:</strong> {(result.next_12m_default_prob * 100).toFixed(1)}%</p>
            <p><strong>12M Prepayment:</strong> {(result.next_12m_prepayment_prob * 100).toFixed(1)}%</p>
            <p><strong>Anomaly Status:</strong> {result.is_anomaly ? <span style={{color: 'red'}}>Flagged</span> : <span style={{color: 'green'}}>Normal</span>}</p>
          </div>
          
          <div style={{ border: '1px solid #ddd', padding: '1.5rem', borderRadius: '8px', background: '#eef6ff' }}>
            <h3>LLM Reviewer Copilot Summary</h3>
            <p style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{result.reviewer_summary}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
