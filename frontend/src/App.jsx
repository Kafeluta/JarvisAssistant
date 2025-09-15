import './App.css'
import { useState } from "react";
import axios from "axios";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const askJarvis = async () => {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/ask", {
        question,
        top_k: 5,
      });
      setAnswer(res.data.answer || "No answer");
    } catch (err) {
      setAnswer("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const ingestDocs = async () => {
    setLoading(true);
    try {
      await axios.post("http://127.0.0.1:8000/ingest", { folder: "./Docs" });
      alert("Docs ingested successfully!");
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ fontSize: "2rem", marginBottom: "1rem" }}>J.A.R.V.I.S. Assistant</h1>
      <button onClick={ingestDocs} disabled={loading}>
        Ingest Docs
      </button>
      <div style={{ marginTop: "1rem" }}>
        <input
          style={{ width: "300px", marginRight: "10px" }}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask JARVIS..."
        />
        <button onClick={askJarvis} disabled={loading}>
          Ask
        </button>
      </div>
      <div style={{ marginTop: "2rem", whiteSpace: "pre-wrap" }}>
        {loading ? "Loading..." : answer}
      </div>
    </div>
  );
}

export default App;
