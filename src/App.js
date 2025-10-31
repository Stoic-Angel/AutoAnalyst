import React, { useState, useRef } from "react";
import { Bar, Pie, Line } from "react-chartjs-2";
import "chart.js/auto";

export default function App() {
  const fileRef = useRef(null);
  const [fileName, setFileName] = useState("");
  const [showDashboard, setShowDashboard] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      setTimeout(() => setShowDashboard(true), 1500); // simulate processing delay
    }
  };

  const triggerUpload = () => fileRef.current.click();

  const barData = { labels: [], datasets: [{ data: [] }] };
  const pieData = { labels: [], datasets: [{ data: [] }] };
  const lineData = { labels: [], datasets: [{ data: [] }] };

  return (
    <div className="min-h-screen flex flex-col items-center px-6 py-10">
      <header className="w-full max-w-5xl flex items-center justify-between mb-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-gradient-to-br from-neon to-teal text-white font-bold shadow-lg">
            AI
          </div>
          <h1 className="text-xl font-semibold">AI Data Analytics Dashboard</h1>
        </div>
        <p className="text-sm text-gray-400">Offline | Secure | Smart</p>
      </header>

      <div className="w-full max-w-5xl glass p-8">
        {!showDashboard ? (
          <div className="flex flex-col items-center justify-center space-y-6">
            <div
              className="w-full md:w-2/3 border-2 border-dashed border-indigo-500/40 p-12 rounded-xl text-center cursor-pointer transition hover:border-indigo-400 hover:shadow-lg"
              onClick={triggerUpload}
            >
              <div className="text-5xl mb-4">📄</div>
              <h2 className="text-lg font-medium text-indigo-300">
                Drag and drop or click to upload PDF
              </h2>
              <p className="text-sm text-gray-400 mt-1">
                Backend team can connect this upload input to /api/upload
              </p>
            </div>

            <input
              ref={fileRef}
              type="file"
              accept=".pdf"
              hidden
              onChange={handleFileChange}
            />

            {fileName && (
              <p className="text-sm text-gray-300">
                Uploaded: <span className="text-indigo-300">{fileName}</span>
              </p>
            )}
          </div>
        ) : (
          <div>
            <h2 className="text-2xl font-semibold text-center mb-10 text-indigo-300">
              Dashboard Insights
            </h2>

            <div className="grid md:grid-cols-3 gap-6 mb-10">
              <div className="glass p-5 text-center">
                <h3 className="text-sm text-gray-400">File Processed</h3>
                <p
                  className="text-xl mt-2 font-semibold truncate max-w-[200px] mx-auto"
                  title={fileName}
                >
                  {fileName
                    ? fileName.length > 25
                      ? fileName.slice(0, 25) + "..."
                      : fileName
                    : "Report.pdf"}
                </p>
              </div>
              <div className="glass p-5 text-center">
                <h3 className="text-sm text-gray-400">AI Confidence</h3>
                <p className="text-xl mt-2 font-semibold">—%</p>
              </div>
              <div className="glass p-5 text-center">
                <h3 className="text-sm text-gray-400">KPIs Extracted</h3>
                <p className="text-xl mt-2 font-semibold">—</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="glass p-4">
                <h3 className="text-sm text-gray-400 mb-2">Bar Chart (KPI Trend)</h3>
                <Bar data={barData} />
              </div>

              <div className="glass p-4">
                <h3 className="text-sm text-gray-400 mb-2">Pie Chart (Category Distribution)</h3>
                <Pie data={pieData} />
              </div>

              <div className="md:col-span-2 glass p-4">
                <h3 className="text-sm text-gray-400 mb-2">Line Chart (Performance Over Time)</h3>
                <Line data={lineData} />
              </div>
            </div>

            <div className="flex justify-center mt-10">
              <button
                onClick={() => setShowDashboard(false)}
                className="px-6 py-3 bg-gradient-to-r from-neon to-teal rounded-lg text-black font-semibold shadow-lg hover:shadow-xl transition"
              >
                Upload Another File
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
