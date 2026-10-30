import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  Factory,
  Gauge,
  History,
  LayoutDashboard,
  Menu,
  Moon,
  Sun,
  Wrench,
  X,
} from "lucide-react";
import {
  Link,
  NavLink,
  Outlet,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  createPrediction,
  getHealth,
  getModelInfo,
  getPredictions,
  readableError,
} from "./api.js";

const navigation = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/analysis", label: "Machine analysis", icon: Gauge },
  { to: "/results", label: "Prediction results", icon: Activity },
  { to: "/monitoring", label: "Benchmark monitor", icon: BarChart3 },
  { to: "/history", label: "Prediction history", icon: History },
  { to: "/about", label: "About project", icon: Database },
];

const inputFields = [
  {
    name: "air_temperature",
    label: "Air temperature",
    unit: "K",
    min: 295.3,
    max: 304.5,
    step: 0.1,
  },
  {
    name: "process_temperature",
    label: "Process temperature",
    unit: "K",
    min: 305.7,
    max: 313.8,
    step: 0.1,
  },
  {
    name: "rotational_speed",
    label: "Rotational speed",
    unit: "rpm",
    min: 1168,
    max: 2886,
    step: 1,
  },
  { name: "torque", label: "Torque", unit: "Nm", min: 3.8, max: 76.6, step: 0.1 },
  { name: "tool_wear", label: "Tool wear", unit: "min", min: 0, max: 253, step: 1 },
];

const analysisPresets = [
  {
    id: "benchmark-no-failure",
    label: "No-failure example",
    description: "Actual AI4I record · label 0",
    values: {
      machine_type: "L",
      air_temperature: "300.0",
      process_temperature: "309.4",
      rotational_speed: "2010",
      torque: "20.0",
      tool_wear: "55",
    },
  },
  {
    id: "benchmark-failure",
    label: "Failure example",
    description: "Actual AI4I record · label 1",
    values: {
      machine_type: "L",
      air_temperature: "298.9",
      process_temperature: "310.2",
      rotational_speed: "2737",
      torque: "8.8",
      tool_wear: "142",
    },
  },
  {
    id: "benchmark-high-wear",
    label: "High tool wear",
    description: "Actual AI4I record · wear 221 min",
    values: {
      machine_type: "L",
      air_temperature: "297.6",
      process_temperature: "309.2",
      rotational_speed: "1390",
      torque: "43.3",
      tool_wear: "221",
    },
  },
];

const palette = ["#5bd5e7", "#ff718b"];

function PageTitle({ eyebrow, title, description, action }) {
  return (
    <div className="page-title">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  );
}

function Loading({ label = "Loading data…" }) {
  return (
    <div className="loading-state" role="status">
      <span className="spinner" />
      {label}
    </div>
  );
}

function ErrorBanner({ message, onRetry }) {
  if (!message) return null;
  return (
    <div className="error-banner" role="alert">
      <AlertTriangle size={18} />
      <span>{message}</span>
      {onRetry && (
        <button className="text-button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

function EmptyState({ title, description, action }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Activity size={21} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}

function StatCard({ label, value, note, icon: Icon, tone = "purple" }) {
  return (
    <article className="stat-card panel">
      <div className={`stat-icon ${tone}`}>
        <Icon size={19} />
      </div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-note">{note}</div>
    </article>
  );
}

function AppLayout() {
  const [theme, setTheme] = useState(() => localStorage.getItem("forge-theme") || "dark");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const location = useLocation();

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("forge-theme", theme);
  }, [theme]);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    getHealth()
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  return (
    <div className="app-shell">
      {mobileOpen && (
        <button
          className="mobile-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <div className="brand-row">
          <Link to="/" className="brand">
            <div className="brand-mark">
              <Factory size={21} />
            </div>
            <div>
              <strong>ForgeSight</strong>
              <span>MAINTENANCE INTELLIGENCE</span>
            </div>
          </Link>
          <button
            className="icon-button mobile-close"
            aria-label="Close menu"
            onClick={() => setMobileOpen(false)}
          >
            <X size={18} />
          </button>
        </div>
        <div className="nav-caption">WORKSPACE</div>
        <nav className="main-nav" aria-label="Main navigation">
          {navigation.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
            >
              <Icon size={18} />
              <span>{label}</span>
              {label === "Prediction results" && <span className="nav-dot" />}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="dataset-badge">
            <div className="dataset-badge-icon">
              <Database size={16} />
            </div>
            <div>
              <strong>AI4I 2020 benchmark</strong>
              <span>Public educational dataset</span>
            </div>
          </div>
          <div className="sidebar-foot">
            <span className={`status-dot ${apiStatus}`} />
            <span>API {apiStatus === "checking" ? "checking" : apiStatus}</span>
            <span className="api-label">LOCAL DEMO</span>
          </div>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            aria-label="Open menu"
            onClick={() => setMobileOpen(true)}
          >
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            <span>Workspace</span>
            <ChevronRight size={14} />
            <strong>{navigation.find((item) => item.to === location.pathname)?.label || "ForgeSight"}</strong>
          </div>
          <div className="topbar-right">
            <div className="prototype-label">
              <span className="prototype-pulse" />
              EDUCATIONAL PROTOTYPE
            </div>
            <button
              className="icon-button theme-toggle"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            >
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <div className="avatar">FS</div>
          </div>
        </header>
        <main className="page-content">
          <Outlet />
        </main>
        <footer className="global-disclaimer">
          <AlertTriangle size={14} />
          <span>
            Educational benchmark demo only — not connected to industrial
            equipment or real-time sensors.
          </span>
        </footer>
      </div>
    </div>
  );
}

function DashboardPage() {
  const [info, setInfo] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const load = () => {
    setLoading(true);
    setError("");
    Promise.all([getModelInfo(), getPredictions(6)])
      .then(([model, records]) => {
        setInfo(model);
        setHistory(records);
      })
      .catch((requestError) => setError(readableError(requestError)))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  const distribution = useMemo(() => {
    if (!info) return [];
    return [
      { name: "No failure", value: info.dataset_target_counts["0"] },
      { name: "Failure", value: info.dataset_target_counts["1"] },
    ];
  }, [info]);

  if (loading) return <Loading label="Loading benchmark and prediction history…" />;
  return (
    <>
      <PageTitle
        eyebrow="SYSTEM OVERVIEW"
        title="Predictive maintenance"
        description="A data-backed view of the UCI benchmark and your saved demo predictions."
        action={
          <Link className="button button-primary" to="/analysis">
            <Activity size={17} /> Run analysis
          </Link>
        }
      />
      <ErrorBanner message={error} onRetry={load} />
      {info && (
        <>
          <div className="stat-grid">
            <StatCard
              label="Benchmark records"
              value={info.dataset_rows.toLocaleString()}
              note="UCI AI4I dataset · offline sample"
              icon={Database}
              tone="cyan"
            />
            <StatCard
              label="Failure-labelled records"
              value={info.dataset_target_counts["1"].toLocaleString()}
              note={`${((info.dataset_target_counts["1"] / info.dataset_rows) * 100).toFixed(2)}% of benchmark`}
              icon={AlertTriangle}
              tone="rose"
            />
            <StatCard
              label="Held-out model recall"
              value={`${(info.test_metrics.recall * 100).toFixed(1)}%`}
              note="Random Forest · default threshold 0.50"
              icon={CheckCircle2}
              tone="green"
            />
            <StatCard
              label="Saved demo predictions"
              value={info.prediction_history_count.toLocaleString()}
              note="All records · local SQLite API history"
              icon={History}
              tone="purple"
            />
          </div>

          <div className="dashboard-grid">
            <section className="panel chart-panel">
              <div className="panel-heading">
                <div>
                  <div className="eyebrow">DATASET LABELS</div>
                  <h2>AI4I failure distribution</h2>
                </div>
                <span className="legend-chip"><i className="cyan-dot" /> Benchmark</span>
              </div>
              <div className="distribution-layout">
                <div className="donut-wrap">
                  <ResponsiveContainer width="100%" height={235}>
                    <PieChart>
                      <Pie
                        data={distribution}
                        dataKey="value"
                        nameKey="name"
                        innerRadius={69}
                        outerRadius={94}
                        paddingAngle={4}
                        stroke="none"
                      >
                        {distribution.map((item, index) => (
                          <Cell key={item.name} fill={palette[index]} />
                        ))}
                      </Pie>
                      <Tooltip
                        formatter={(value) => [Number(value).toLocaleString(), "Records"]}
                        contentStyle={{ background: "var(--tooltip)", borderColor: "var(--border)" }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="donut-center">
                    <strong>{info.dataset_rows.toLocaleString()}</strong>
                    <span>records</span>
                  </div>
                </div>
                <div className="distribution-legend">
                  {distribution.map((item, index) => (
                    <div className="distribution-item" key={item.name}>
                      <span className="legend-dot" style={{ background: palette[index] }} />
                      <div><span>{item.name}</span><strong>{item.value.toLocaleString()}</strong></div>
                      <small>{((item.value / info.dataset_rows) * 100).toFixed(2)}%</small>
                    </div>
                  ))}
                  <p className="chart-note">Observed target labels in the public dataset—not live machine health.</p>
                </div>
              </div>
            </section>

            <section className="panel model-panel">
              <div className="panel-heading">
                <div>
                  <div className="eyebrow">HELD-OUT TEST</div>
                  <h2>{info.model} performance</h2>
                </div>
                <span className="small-pill">AI4I benchmark</span>
              </div>
              <div className="metric-bars">
                {[
                  ["Precision", info.test_metrics.precision, "#8b7cff"],
                  ["Recall", info.test_metrics.recall, "#59d5e7"],
                  ["F1 score", info.test_metrics.f1, "#f5b75b"],
                  ["ROC-AUC", info.test_metrics.roc_auc, "#69d69b"],
                  ["PR-AUC", info.test_metrics.pr_auc_average_precision, "#ff718b"],
                ].map(([label, value, color]) => (
                  <div className="metric-bar" key={label}>
                    <div><span>{label}</span><strong>{(value * 100).toFixed(1)}%</strong></div>
                    <div className="progress-track"><span style={{ width: `${value * 100}%`, background: color }} /></div>
                  </div>
                ))}
              </div>
              <p className="chart-note">
                Recall is the share of failure labels detected at threshold 0.50.
                Model scores are not calibrated probabilities.
              </p>
            </section>
          </div>

          <section className="panel recent-panel">
            <div className="panel-heading">
              <div>
                <div className="eyebrow">LOCAL SQLITE HISTORY</div>
                <h2>Recent predictions</h2>
              </div>
              <Link to="/history" className="link-button">View history <ChevronRight size={15} /></Link>
            </div>
            {!history.length ? (
              <EmptyState
                title="No predictions saved yet"
                description="Submit benchmark parameters to create your first prediction record."
                action={<Link className="button button-secondary" to="/analysis">Start an analysis</Link>}
              />
            ) : <PredictionTable records={history.slice(0, 5)} compact />}
          </section>
        </>
      )}
    </>
  );
}

function AnalysisPage() {
  const navigate = useNavigate();
  const [values, setValues] = useState({
    machine_type: "L",
    air_temperature: "",
    process_temperature: "",
    rotational_speed: "",
    torque: "",
    tool_wear: "",
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function applyPreset(preset) {
    setValues(preset.values);
    setError("");
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const payload = {
        ...values,
        air_temperature: Number(values.air_temperature),
        process_temperature: Number(values.process_temperature),
        rotational_speed: Number(values.rotational_speed),
        torque: Number(values.torque),
        tool_wear: Number(values.tool_wear),
      };
      const result = await createPrediction(payload);
      navigate("/results", { state: { prediction: result } });
    } catch (requestError) {
      setError(readableError(requestError));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageTitle
        eyebrow="MODEL INFERENCE"
        title="Machine analysis"
        description="Enter a single AI4I-style parameter record to request a prediction from the FastAPI model."
      />
      <div className="analysis-layout">
        <form className="panel form-panel" onSubmit={submit}>
          <div className="panel-heading">
            <div><div className="eyebrow">INPUT PARAMETERS</div><h2>Operating conditions</h2></div>
            <span className="small-pill"><span className="required-dot" /> Required</span>
          </div>
          <ErrorBanner message={error} />
          <div className="preset-section">
            <div className="preset-heading">
              <span>QUICK FILL</span>
              <small>Actual examples from the AI4I dataset</small>
            </div>
            <div className="preset-list">
              {analysisPresets.map((preset) => (
                <button
                  className="preset-button"
                  key={preset.id}
                  type="button"
                  onClick={() => applyPreset(preset)}
                  disabled={busy}
                  aria-label={`Fill form with ${preset.label}: ${preset.description}`}
                >
                  <strong>{preset.label}</strong>
                  <span>{preset.description}</span>
                </button>
              ))}
            </div>
            <p className="preset-note">
              Dataset labels describe the original records; the model prediction
              may differ. Presets only fill the form and do not submit it.
            </p>
          </div>
          <label className="field">
            <span>Machine type <span className="unit">AI4I product category</span></span>
            <select
              value={values.machine_type}
              onChange={(event) => setValues({ ...values, machine_type: event.target.value })}
              required
            >
              <option value="L">L · Low</option>
              <option value="M">M · Medium</option>
              <option value="H">H · High</option>
            </select>
          </label>
          <div className="form-grid">
            {inputFields.map((field) => (
              <label className="field" key={field.name}>
                <span>{field.label} <span className="unit">{field.unit}</span></span>
                <input
                  type="number"
                  name={field.name}
                  value={values[field.name]}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  placeholder={`${field.min} – ${field.max}`}
                  required
                  onChange={(event) => setValues({ ...values, [field.name]: event.target.value })}
                />
                <small>Benchmark range: {field.min}–{field.max} {field.unit}</small>
              </label>
            ))}
          </div>
          <div className="form-disclaimer">
            <Database size={16} />
            <span>Inputs are restricted to values observed in AI4I 2020. Results are educational and not an equipment diagnosis.</span>
          </div>
          <button className="button button-primary submit-button" type="submit" disabled={busy}>
            {busy ? <><span className="button-spinner" /> Evaluating model…</> : <><Activity size={17} /> Generate prediction</>}
          </button>
        </form>

        <aside className="analysis-aside">
          <div className="panel info-panel">
            <div className="aside-icon"><Cpu size={20} /></div>
            <div className="eyebrow">MODEL PIPELINE</div>
            <h3>One request, one model score</h3>
            <p>Your parameters are processed by the saved Random Forest pipeline and sent to the local API.</p>
            <div className="pipeline-steps">
              {["Validate AI4I parameter ranges", "Transform with saved preprocessing", "Estimate model score", "Save result to local history"].map((item, index) => (
                <div className="pipeline-step" key={item}><span>{String(index + 1).padStart(2, "0")}</span>{item}</div>
              ))}
            </div>
          </div>
          <div className="notice-card">
            <AlertTriangle size={18} />
            <div><strong>Not a live monitoring system</strong><p>No sensor feed is connected. Predictions apply only to the supplied benchmark-style record.</p></div>
          </div>
        </aside>
      </div>
    </>
  );
}

function RiskPill({ risk }) {
  return <span className={`risk-pill risk-${(risk || "low").toLowerCase()}`}><i />{risk || "Unknown"} risk</span>;
}

function ResultCard({ prediction }) {
  if (!prediction) return null;
  const failure = prediction.predicted_failure === 1;
  const percentage = (prediction.failure_probability * 100).toFixed(1);
  return (
    <div className="result-layout">
      <section className={`panel result-hero ${failure ? "result-warning" : "result-ok"}`}>
        <div className="result-status-icon">{failure ? <AlertTriangle size={25} /> : <CheckCircle2 size={25} />}</div>
        <div className="eyebrow">MODEL CLASSIFICATION</div>
        <h2>{failure ? "Failure predicted" : "No failure predicted"}</h2>
        <p>Classifier decision at its documented 0.50 threshold.</p>
        <div className="result-score">
          <strong>{percentage}<small>%</small></strong>
          <span>model-estimated failure score</span>
        </div>
        <RiskPill risk={prediction.risk_category} />
        <div className="score-track"><span style={{ width: `${Math.min(prediction.failure_probability * 100, 100)}%` }} /></div>
        <div className="score-boundaries"><span>0%</span><span>100%</span></div>
      </section>

      <section className="panel recommendation-panel">
        <div className="panel-heading"><div><div className="eyebrow">RULE-BASED GUIDANCE</div><h2>Maintenance recommendations</h2></div><Wrench size={19} className="heading-icon" /></div>
        <p className="section-subtitle">These checks are separate from the machine-learning prediction.</p>
        <div className="recommendation-list">
          {prediction.recommendations.map((item, index) => (
            <div className="recommendation-item" key={item}>
              <span>{String(index + 1).padStart(2, "0")}</span><p>{item}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="panel parameters-panel">
        <div className="panel-heading"><div><div className="eyebrow">SUBMITTED RECORD</div><h2>Parameter summary</h2></div><Database size={18} className="heading-icon" /></div>
        <div className="parameter-grid">
          {[
            ["Machine type", prediction.parameters.Type, ""],
            ["Air temperature", prediction.parameters["Air temperature"], "K"],
            ["Process temperature", prediction.parameters["Process temperature"], "K"],
            ["Rotational speed", prediction.parameters["Rotational speed"], "rpm"],
            ["Torque", prediction.parameters.Torque, "Nm"],
            ["Tool wear", prediction.parameters["Tool wear"], "min"],
          ].map(([label, value, unit]) => (
            <div className="parameter-item" key={label}><span>{label}</span><strong>{value} <small>{unit}</small></strong></div>
          ))}
        </div>
      </section>
      <div className="probability-note"><AlertTriangle size={15} /><span>{prediction.probability_note} Risk bands are heuristic: Low &lt; 10%, Medium 10–&lt;30%, High ≥ 30%.</span></div>
    </div>
  );
}

function ResultsPage() {
  const location = useLocation();
  const [prediction, setPrediction] = useState(location.state?.prediction || null);
  const [loading, setLoading] = useState(!location.state?.prediction);
  const [error, setError] = useState("");
  useEffect(() => {
    if (location.state?.prediction) {
      setPrediction(location.state.prediction);
      setLoading(false);
      return;
    }
    getPredictions(1)
      .then((records) => setPrediction(records[0] || null))
      .catch((requestError) => setError(readableError(requestError)))
      .finally(() => setLoading(false));
  }, [location.state]);

  if (loading) return <Loading label="Loading the latest prediction…" />;
  return (
    <>
      <PageTitle eyebrow="PREDICTION DETAIL" title="Prediction results" description="Model output and clearly separated rule-based recommendations." action={<Link to="/analysis" className="button button-primary"><Activity size={17} /> New analysis</Link>} />
      <ErrorBanner message={error} />
      {prediction ? <ResultCard prediction={prediction} /> : <EmptyState title="No prediction to display" description="Run a machine analysis first. Results will be saved by the backend." action={<Link className="button button-primary" to="/analysis">Open machine analysis</Link>} />}
    </>
  );
}

function MonitoringPage() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    getModelInfo().then(setInfo).catch((requestError) => setError(readableError(requestError)));
  }, []);
  const rangeData = useMemo(() => info ? Object.entries(info.supported_input_ranges).map(([name, range]) => ({
    name: name.replace(" temperature", " temp."),
    label: `${range.minimum}–${range.maximum}`,
  })) : [], [info]);

  return (
    <>
      <PageTitle eyebrow="DATASET REFERENCE" title="Benchmark monitor" description="Observed input ranges in the AI4I benchmark. This page does not display live machine telemetry." />
      <ErrorBanner message={error} />
      {!info ? <Loading label="Loading benchmark ranges…" /> : (
        <>
          <div className="notice-card full-notice"><Database size={18} /><div><strong>Reference data, not sensor readings</strong>          <p>Charts show distributions from the public benchmark records. They are not a time series and do not indicate current machine condition.</p></div></div>
          <section className="panel range-panel">
            <div className="panel-heading"><div><div className="eyebrow">OBSERVED BENCHMARK DISTRIBUTIONS</div><h2>Operating parameter samples</h2></div><span className="small-pill">AI4I · {info.dataset_rows.toLocaleString()} records</span></div>
            <div className="monitoring-grid">
          {Object.entries(info.parameter_distributions).map(([name, bins], index) => (
            <article className="distribution-chart-card" key={name}>
              <div className="distribution-chart-heading">
                <strong>{name}</strong>
                <span>{rangeData[index]?.label} {name.includes("temperature") ? "K" : name === "Torque" ? "Nm" : name === "Tool wear" ? "min" : "rpm"}</span>
              </div>
              <ResponsiveContainer width="100%" height={205}>
                <BarChart data={bins} margin={{ left: -18, right: 7, top: 12, bottom: 25 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="range" tick={{ fill: "var(--muted)", fontSize: 8 }} angle={-30} textAnchor="end" interval={0} />
                  <YAxis tick={{ fill: "var(--muted)", fontSize: 8 }} width={35} />
                  <Tooltip formatter={(value) => [Number(value).toLocaleString(), "Records"]} labelFormatter={(label) => `${name}: ${label}`} contentStyle={{ background: "var(--tooltip)", borderColor: "var(--border)", borderRadius: 10, fontSize: 10 }} />
                  <Bar dataKey="count" name="Records" fill={index % 2 ? "#5bd5e7" : "#8b7cff"} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </article>
          ))}
            </div>
            <div className="range-list">
          {Object.entries(info.supported_input_ranges).map(([name, range]) => <div key={name}><span>{name}</span><strong>{range.minimum}–{range.maximum}</strong></div>)}
            </div>
          </section>
          <section className="panel model-note-panel">
            <div className="aside-icon"><Gauge size={19} /></div>
            <div><h3>How to read this view</h3><p>The benchmark contains samples of machine type, air and process temperature, rotational speed, torque, and tool wear. The range is only a data coverage summary, not an operating safety limit.</p></div>
          </section>
        </>
      )}
    </>
  );
}

function PredictionTable({ records, compact = false }) {
  if (!records?.length) return <EmptyState title="No saved predictions" description="Predictions created through the analysis form will appear here." />;
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead><tr><th>Record</th><th>Date & time</th><th>Machine type</th><th>Prediction</th><th>Model score</th><th>Risk</th></tr></thead>
        <tbody>
          {records.map((record) => (
            <tr key={record.id}>
              <td><span className="record-id">#{record.id}</span></td>
              <td>{new Date(record.created_at).toLocaleString()}</td>
              <td><span className="type-chip">{record.parameters?.Type || "—"}</span></td>
              <td><span className={`prediction-label ${record.predicted_failure ? "is-failure" : "is-ok"}`}><i />{record.predicted_failure ? "Failure" : "No failure"}</span></td>
              <td className="score-cell">{(record.failure_probability * 100).toFixed(1)}%</td>
              <td><RiskPill risk={record.risk_category} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      {compact && <div className="table-footer">Showing latest {records.length} saved predictions</div>}
    </div>
  );
}

function HistoryPage() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const load = () => {
    setLoading(true);
    setError("");
    getPredictions(100).then(setRecords).catch((requestError) => setError(readableError(requestError))).finally(() => setLoading(false));
  };
  useEffect(load, []);
  return (
    <>
      <PageTitle eyebrow="LOCAL RECORDS" title="Prediction history" description="Previous model predictions saved by this FastAPI instance to its local SQLite database." action={<button className="button button-secondary" onClick={load}><History size={16} /> Refresh</button>} />
      <ErrorBanner message={error} onRetry={load} />
      <section className="panel history-panel">
        <div className="panel-heading"><div><div className="eyebrow">MOST RECENT FIRST</div><h2>Saved predictions</h2></div><span className="small-pill">{records.length} loaded</span></div>
        {loading ? <Loading label="Loading saved predictions…" /> : records.length ? <PredictionTable records={records} /> : <EmptyState title="History is empty" description="Submit a prediction from Machine analysis. The API stores only the entered benchmark parameters and result." action={<Link className="button button-primary" to="/analysis">Start an analysis</Link>} />}
      </section>
    </>
  );
}

function AboutPage() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    getModelInfo().then(setInfo).catch((requestError) => setError(readableError(requestError)));
  }, []);
  return (
    <>
      <PageTitle eyebrow="PROJECT DOCUMENTATION" title="About ForgeSight" description="A student-friendly demonstration of a machine-learning predictive maintenance workflow." />
      <ErrorBanner message={error} />
      <div className="about-grid">
        <section className="panel about-main">
          <div className="about-icon"><Factory size={23} /></div>
          <div className="eyebrow">PROJECT OBJECTIVE</div>
          <h2>Explore failure prediction with a public benchmark</h2>
          <p>ForgeSight demonstrates how machine operating parameters can be passed through a trained classifier to estimate whether a benchmark record is associated with machine failure. It is an educational prototype only—not a live monitoring service and not a validated industrial decision system.</p>
          <div className="about-tags"><span>Classification</span><span>Imbalanced data</span><span>REST API</span><span>React dashboard</span></div>
        </section>
        <section className="panel about-card">
          <div className="eyebrow">DATASET</div><h3>UCI AI4I 2020</h3>
          <p>Public synthetic predictive-maintenance benchmark. The raw local copy has 10,000 rows. Failure mode labels are excluded from model inputs to prevent target leakage.</p>
          <a className="external-link" href="https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset" target="_blank" rel="noreferrer">Dataset page <ChevronRight size={15} /></a>
        </section>
        <section className="panel about-card">
          <div className="eyebrow">METHODOLOGY</div><h3>{info?.model || "Selected model"}</h3>
          <p>Random Forest was selected using training-only stratified cross-validation average precision. Held-out recall and precision-recall metrics are shown because failures are rare. The default 0.50 decision threshold is not tuned for operational use.</p>
          {info && <div className="mini-metrics"><span>Recall <strong>{(info.test_metrics.recall * 100).toFixed(1)}%</strong></span><span>PR-AUC <strong>{info.test_metrics.pr_auc_average_precision.toFixed(3)}</strong></span></div>}
        </section>
        <section className="panel about-card stack-card">
          <div className="eyebrow">TECHNOLOGY STACK</div><h3>Built with familiar tools</h3>
          <div className="stack-list">{["Python · Pandas · Scikit-learn", "FastAPI · SQLite · Joblib", "React · Vite · Tailwind CSS", "Recharts · Axios · Lucide"].map((item) => <div key={item}><CheckCircle2 size={15} />{item}</div>)}</div>
        </section>
        <section className="limitation-card">
          <AlertTriangle size={20} /><div><strong>Important limitations</strong><p>The dataset is not live or field telemetry. The model score is not guaranteed to be calibrated. A false negative could miss a failure; a false positive could prompt unnecessary checks. Do not use this prototype to make operational or safety decisions.</p></div>
        </section>
      </div>
    </>
  );
}

function NotFoundPage() {
  return <EmptyState title="Page not found" description="That route is not part of this demo." action={<Link className="button button-primary" to="/">Return to overview</Link>} />;
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="analysis" element={<AnalysisPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="monitoring" element={<MonitoringPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
