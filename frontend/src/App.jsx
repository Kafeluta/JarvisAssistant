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
    const q = question.trim();
    if (q.toLowerCase().startsWith("look at my screen")) {
      // everything after the phrase becomes the goal (optional)
      const goal = q.slice("look at my screen".length).trim() ||
                   "Critique the design and code on screen; propose concrete fixes.";
      const res = await axios.post("http://127.0.0.1:8000/screen_review", {
        goal,
        top_k: 12,
        target_words: 220,
      });
      setAnswer(res.data.answer || "No advice yet.");
    } else {
      const res = await axios.post("http://127.0.0.1:8000/ask", { question: q, top_k: 5 });
      setAnswer(res.data.answer || "No answer");
    }
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
