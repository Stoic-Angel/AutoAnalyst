import React, { useState, useRef, useMemo } from "react";
import { Bar, Pie, Line, Doughnut } from "react-chartjs-2";
import "chart.js/auto";

const API_URL = "http://localhost:8000";

export default function App() {
  const fileRef = useRef(null);
  const [fileName, setFileName] = useState("");
  const [showDashboard, setShowDashboard] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  
  // Dashboard interactivity states
  const [activeFilter, setActiveFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedSection, setExpandedSection] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      setAnalysisData(data);
      setShowDashboard(true);
    } catch (err) {
      setError(err.message || "Failed to process file. Is the backend running?");
    } finally {
      setIsLoading(false);
    }
  };

  const resetDashboard = () => {
    setShowDashboard(false);
    setAnalysisData(null);
    setFileName("");
    setError(null);
    setActiveFilter("all");
    setSearchTerm("");
  };

  // Upload Screen
  if (!showDashboard) {
    return (
      <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center px-6 py-10">
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-cyan-400 text-white text-2xl font-bold mb-4 shadow-lg shadow-indigo-500/30">
            AI
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">AutoAnalyst</h1>
          <p className="text-slate-400">AI-Powered Data Analytics Dashboard</p>
        </div>

        <div
          className={`w-full max-w-xl border-2 border-dashed rounded-2xl p-16 text-center transition-all cursor-pointer ${
            isLoading ? "border-amber-500/50 bg-amber-500/5" : "border-slate-600 hover:border-indigo-500 hover:bg-slate-800/50"
          }`}
          onClick={!isLoading ? () => fileRef.current.click() : undefined}
        >
          {isLoading ? (
            <>
              <div className="w-16 h-16 mx-auto mb-4 border-4 border-amber-500 border-t-transparent rounded-full animate-spin"></div>
              <h2 className="text-xl font-semibold text-amber-400">Analyzing your data...</h2>
              <p className="text-slate-500 mt-2">Generating insights and visualizations</p>
            </>
          ) : (
            <>
              <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-slate-800 flex items-center justify-center">
                <svg className="w-10 h-10 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-white">Drop your file here</h2>
              <p className="text-slate-400 mt-2">or click to browse</p>
              <p className="text-slate-500 text-sm mt-4">Supports PDF and CSV files</p>
            </>
          )}
        </div>

        <input ref={fileRef} type="file" accept=".pdf,.csv" hidden onChange={handleFileChange} disabled={isLoading} />

        {error && (
          <div className="mt-6 w-full max-w-xl bg-red-500/10 border border-red-500/30 rounded-xl p-4">
            <p className="text-red-400">{error}</p>
          </div>
        )}
      </div>
    );
  }

  // Dashboard Data
  const dashboard = analysisData?.charts || {};
  const kpis = dashboard.kpis || [];
  const trends = dashboard.trends || [];
  const distributions = dashboard.distributions || [];
  const comparisons = dashboard.comparisons || [];
  const insights = dashboard.insights || [];
  const tableData = dashboard.tableData || null;
  const filters = dashboard.filters || [];

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Header */}
      <header className="bg-slate-800/80 backdrop-blur-sm border-b border-slate-700 px-6 py-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 flex items-center justify-center text-white font-bold">
              AI
            </div>
            <div>
              <h1 className="font-semibold">{analysisData?.document_type || "Data Analysis"}</h1>
              <p className="text-sm text-slate-400">{fileName}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* Filter Dropdown */}
            {filters.length > 0 && (
              <select
                value={activeFilter}
                onChange={(e) => setActiveFilter(e.target.value)}
                className="px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-sm focus:outline-none focus:border-indigo-500"
              >
                <option value="all">All Data</option>
                {filters.map((f, i) => (
                  <option key={i} value={f.value}>{f.label}</option>
                ))}
              </select>
            )}
            <button
              onClick={resetDashboard}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm font-medium transition"
            >
              New Analysis
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        
        {/* SECTION 1: KPI Layer - Current State */}
        <section>
          <SectionHeader 
            title="Key Performance Indicators" 
            subtitle="Current state at a glance"
            icon="📊"
          />
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {kpis.map((kpi, i) => (
              <KPICard key={i} {...kpi} />
            ))}
          </div>
        </section>

        {/* SECTION 2: Actionable Insights */}
        {insights.length > 0 && (
          <section>
            <SectionHeader 
              title="Key Insights" 
              subtitle="AI-generated observations and recommendations"
              icon="💡"
            />
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {insights.map((insight, i) => (
                <InsightCard key={i} insight={insight} index={i} />
              ))}
            </div>
          </section>
        )}

        {/* SECTION 3: Trend Analysis - Time-Based */}
        {trends.length > 0 && (
          <section>
            <SectionHeader 
              title="Trend Analysis" 
              subtitle="Performance over time"
              icon="📈"
            />
            <div className="grid lg:grid-cols-2 gap-6">
              {trends.map((trend, i) => (
                <ChartCard key={i} visualization={trend} expandable onExpand={() => setExpandedSection(`trend-${i}`)} />
              ))}
            </div>
          </section>
        )}

        {/* SECTION 4: Distribution Analysis */}
        {distributions.length > 0 && (
          <section>
            <SectionHeader 
              title="Distribution Analysis" 
              subtitle="Breakdown by categories"
              icon="📊"
            />
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {distributions.map((dist, i) => (
                <ChartCard key={i} visualization={dist} expandable onExpand={() => setExpandedSection(`dist-${i}`)} />
              ))}
            </div>
          </section>
        )}

        {/* SECTION 5: Comparison Analysis */}
        {comparisons.length > 0 && (
          <section>
            <SectionHeader 
              title="Comparison Analysis" 
              subtitle="Current vs Previous / Category comparisons"
              icon="⚖️"
            />
            <div className="grid lg:grid-cols-2 gap-6">
              {comparisons.map((comp, i) => (
                <ComparisonCard key={i} data={comp} />
              ))}
            </div>
          </section>
        )}

        {/* SECTION 6: Data Table with Search/Filter/Export */}
        {tableData && (
          <section>
            <SectionHeader 
              title="Data Explorer" 
              subtitle="Raw data with search and export"
              icon="📋"
            />
            <DataTable 
              data={tableData} 
              searchTerm={searchTerm} 
              onSearchChange={setSearchTerm}
            />
          </section>
        )}

        {/* Statistics Summary */}
        {analysisData?.numeric_stats && Object.keys(analysisData.numeric_stats).length > 0 && (
          <section>
            <SectionHeader 
              title="Statistical Summary" 
              subtitle="Numeric field analysis"
              icon="🔢"
            />
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(analysisData.numeric_stats).slice(0, 8).map(([col, stats]) => (
                <StatCard key={col} column={col} stats={stats} />
              ))}
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-6 mt-12">
        <p className="text-center text-slate-500 text-sm">
          AutoAnalyst • AI-Powered Data Analytics
        </p>
      </footer>
    </div>
  );
}

// ============ COMPONENTS ============

function SectionHeader({ title, subtitle, icon }) {
  return (
    <div className="mb-4">
      <h2 className="text-lg font-semibold flex items-center gap-2">
        <span>{icon}</span>
        {title}
      </h2>
      <p className="text-sm text-slate-400">{subtitle}</p>
    </div>
  );
}

function KPICard({ label, value, change, changeType, status, prefix = "", suffix = "", sparkline, description, formatHint }) {
  const isPositive = changeType === "positive" || (typeof change === "number" && change > 0);
  const isNegative = changeType === "negative" || (typeof change === "number" && change < 0);
  
  const statusColors = {
    good: "border-l-emerald-500",
    bad: "border-l-rose-500",
    warning: "border-l-amber-500",
    neutral: "border-l-slate-500"
  };

  const sparklineData = sparkline?.length > 0 ? {
    labels: sparkline.map((_, i) => i),
    datasets: [{
      data: sparkline,
      borderColor: isNegative ? "#f43f5e" : "#10b981",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.4,
      fill: false
    }]
  } : null;

  return (
    <div className={`bg-slate-800/50 rounded-xl border border-slate-700 border-l-4 ${statusColors[status] || statusColors.neutral} p-4 hover:bg-slate-800 transition`}>
      <p className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-1 truncate">{label}</p>
      <div className="flex items-end justify-between gap-2">
        <p className="text-2xl font-bold truncate">
          {prefix}{typeof value === "number" ? formatNumber(value, formatHint) : value}{suffix}
        </p>
        {sparklineData && (
          <div className="w-16 h-8 flex-shrink-0">
            <Line data={sparklineData} options={sparklineOptions} />
          </div>
        )}
      </div>
      {change != null && (
        <div className={`flex items-center gap-1 mt-2 text-sm ${isPositive ? "text-emerald-400" : isNegative ? "text-rose-400" : "text-slate-400"}`}>
          <span>{isPositive ? "↑" : isNegative ? "↓" : "→"}</span>
          <span>{typeof change === "number" ? `${Math.abs(change).toFixed(1)}%` : change}</span>
          {description && <span className="text-slate-500 text-xs ml-1">vs prev</span>}
        </div>
      )}
    </div>
  );
}

function InsightCard({ insight, index }) {
  const priorities = ["high", "medium", "low"];
  const priority = insight.priority || priorities[index % 3];
  
  const priorityStyles = {
    high: "border-l-rose-500 bg-rose-500/5",
    medium: "border-l-amber-500 bg-amber-500/5",
    low: "border-l-emerald-500 bg-emerald-500/5"
  };

  const icon = insight.icon || (insight.text || insight).slice(0, 2);
  const text = insight.text || (typeof insight === 'string' ? insight.slice(2) : insight);

  return (
    <div className={`rounded-xl border border-slate-700 border-l-4 ${priorityStyles[priority]} p-4`}>
      <div className="flex gap-3">
        <span className="text-2xl">{icon}</span>
        <div>
          <p className="text-slate-200">{text}</p>
          {insight.action && (
            <p className="text-sm text-indigo-400 mt-2 font-medium">→ {insight.action}</p>
          )}
        </div>
      </div>
    </div>
  );
}

function ChartCard({ visualization, expandable, onExpand }) {
  const { type, title, description, data } = visualization;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { 
        display: type === "pie" || type === "doughnut",
        position: "bottom",
        labels: { color: "#94a3b8", padding: 15, usePointStyle: true }
      }
    },
    scales: type === "pie" || type === "doughnut" ? {} : {
      x: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" } },
      y: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" } }
    },
    onClick: expandable ? onExpand : undefined
  };

  const renderChart = () => {
    if (!data) return <div className="text-slate-500 text-center py-8">No data available</div>;
    switch (type) {
      case "bar": return <Bar data={data} options={chartOptions} />;
      case "line": return <Line data={data} options={chartOptions} />;
      case "pie": return <Pie data={data} options={chartOptions} />;
      case "doughnut": return <Doughnut data={data} options={chartOptions} />;
      default: return <Bar data={data} options={chartOptions} />;
    }
  };

  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-5 hover:border-slate-600 transition">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-semibold text-white">{title}</h3>
          {description && <p className="text-sm text-slate-400 mt-1">{description}</p>}
        </div>
        {expandable && (
          <button onClick={onExpand} className="text-slate-400 hover:text-white p-1">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
        )}
      </div>
      <div className="aspect-[4/3]">{renderChart()}</div>
    </div>
  );
}

function ComparisonCard({ data }) {
  const { title, current, previous, change, items } = data;
  const isPositive = change > 0;

  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-5">
      <h3 className="font-semibold mb-4">{title}</h3>
      
      {current !== undefined && previous !== undefined ? (
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <p className="text-sm text-slate-400 mb-1">Current</p>
            <p className="text-2xl font-bold text-white">{formatNumber(current)}</p>
          </div>
          <div className="bg-slate-700/50 rounded-lg p-4 text-center">
            <p className="text-sm text-slate-400 mb-1">Previous</p>
            <p className="text-2xl font-bold text-slate-400">{formatNumber(previous)}</p>
          </div>
        </div>
      ) : null}
      
      {change !== undefined && (
        <div className={`text-center py-2 rounded-lg ${isPositive ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"}`}>
          <span className="font-semibold">{isPositive ? "↑" : "↓"} {Math.abs(change).toFixed(1)}%</span>
          <span className="text-sm ml-2">change</span>
        </div>
      )}

      {items && (
        <div className="space-y-2 mt-4">
          {items.map((item, i) => (
            <div key={i} className="flex items-center justify-between py-2 border-b border-slate-700 last:border-0">
              <span className="text-slate-300">{item.label}</span>
              <div className="flex items-center gap-2">
                <span className="font-semibold">{formatNumber(item.value)}</span>
                {item.change && (
                  <span className={`text-sm ${item.change > 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {item.change > 0 ? "+" : ""}{item.change.toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function DataTable({ data, searchTerm, onSearchChange }) {
  const { headers, rows, title } = data;
  
  const filteredRows = useMemo(() => {
    if (!searchTerm) return rows;
    return rows.filter(row => 
      row.some(cell => 
        String(cell).toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
  }, [rows, searchTerm]);

  const exportCSV = () => {
    const csvContent = [
      headers.join(","),
      ...filteredRows.map(row => row.map(cell => `"${cell}"`).join(","))
    ].join("\n");
    
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "data_export.csv";
    a.click();
  };

  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700 overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-700 flex items-center justify-between gap-4">
        <div className="flex-1 max-w-md">
          <input
            type="text"
            placeholder="Search data..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-sm focus:outline-none focus:border-indigo-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-400">{filteredRows.length} rows</span>
          <button
            onClick={exportCSV}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-medium transition flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export
          </button>
        </div>
      </div>
      <div className="overflow-x-auto max-h-96">
        <table className="w-full text-sm">
          <thead className="bg-slate-700/50 sticky top-0">
            <tr>
              {headers?.map((h, i) => (
                <th key={i} className="px-4 py-3 text-left font-medium text-slate-300 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredRows?.slice(0, 50).map((row, i) => (
              <tr key={i} className="border-t border-slate-700/50 hover:bg-slate-700/30">
                {row.map((cell, j) => (
                  <td key={j} className="px-4 py-3 text-slate-300 whitespace-nowrap">{formatCell(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {filteredRows.length > 50 && (
        <div className="px-5 py-3 border-t border-slate-700 text-center text-sm text-slate-400">
          Showing 50 of {filteredRows.length} rows
        </div>
      )}
    </div>
  );
}

function StatCard({ column, stats }) {
  const isYearAxis = /\byear\b/i.test(column) || stats.median != null;
  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-4">
      <h4 className="text-sm font-medium text-slate-400 truncate mb-3">{column}</h4>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <span className="text-xs text-slate-500">Mean</span>
          <p className="font-semibold text-white">{formatNumber(stats.mean, isYearAxis ? "year" : null)}</p>
        </div>
        {stats.sum != null && (
          <div>
            <span className="text-xs text-slate-500">Sum</span>
            <p className="font-semibold text-white">{formatNumber(stats.sum)}</p>
          </div>
        )}
        {stats.median != null && (
          <div>
            <span className="text-xs text-slate-500">Median</span>
            <p className="font-semibold text-white">{formatNumber(stats.median, "year")}</p>
          </div>
        )}
        <div>
          <span className="text-xs text-slate-500">Min</span>
          <p className="font-semibold text-emerald-400">{formatNumber(stats.min, isYearAxis ? "year" : null)}</p>
        </div>
        <div>
          <span className="text-xs text-slate-500">Max</span>
          <p className="font-semibold text-rose-400">{formatNumber(stats.max, isYearAxis ? "year" : null)}</p>
        </div>
      </div>
    </div>
  );
}

// ============ UTILITIES ============

const sparklineOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false }, tooltip: { enabled: false } },
  scales: { x: { display: false }, y: { display: false } }
};

function formatNumber(num, hint) {
  if (num === null || num === undefined || isNaN(num)) return "—";
  if (hint === "year") {
    return String(Math.round(num));
  }
  if (Math.abs(num) >= 1e9) return (num / 1e9).toFixed(1) + "B";
  if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(1) + "M";
  if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(1) + "K";
  if (Number.isInteger(num)) return num.toLocaleString();
  return num.toFixed(2);
}

function formatCell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (value >= 1800 && value <= 2105 && Number.isFinite(value))
      return String(Math.round(value));
    return formatNumber(value);
  }
  return String(value).slice(0, 50);
}
