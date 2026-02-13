"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
    ArrowLeft,
    Activity,
    Cpu,
    HardDrive,
    Thermometer,
    Zap,
    TrendingDown,
    Clock,
    DollarSign,
    Wifi,
    WifiOff,
    BarChart3,
    Download,
    CheckCircle2,
    Shield,
    AlertTriangle,
} from "lucide-react";

// ── VM Config ──
// Uses /api/metrics proxy route (avoids mixed-content HTTPS→HTTP on Vercel)
const METRICS_URL = `/api/metrics`;
const POLL_INTERVAL = 10_000;
const SPOT_PRICE = 0.22;

interface DownloadableFile {
    name: string;
    size_bytes: number;
    size_human: string;
}

interface Metrics {
    status: string;
    gpu_name: string;
    gpu_temp: number;
    gpu_util: number;
    gpu_mem_used: number;
    gpu_mem_total: number;
    current_epoch: number;
    total_epochs: number;
    train_loss: number;
    valid_loss: number;
    best_loss: number;
    lr: number;
    disk_used: string;
    disk_total: string;
    disk_pct: number;
    history: Array<{ train: number; valid: number; lr?: number }>;
    log_tail: string[];
    dataset_progress: string;
    training_complete: boolean;
    downloadable_files: DownloadableFile[];
    watchdog: {
        status: string;
        alerts: Array<{ time: string; level: string; message: string; action_taken?: string }>;
        auto_actions: Array<{ time: string; level: string; message: string; action_taken?: string }>;
        last_check: string;
    };
}

function StatusDot({ status }: { status: string }) {
    const color =
        status === "training"
            ? "bg-emerald-400 shadow-emerald-400/50"
            : status.includes("download") || status.includes("generat")
                ? "bg-amber-400 shadow-amber-400/50"
                : status === "waiting" || status === "unknown"
                    ? "bg-zinc-500"
                    : "bg-red-400 shadow-red-400/50";
    return (
        <span
            className={`inline-block w-2 h-2 rounded-full ${color} shadow-lg animate-pulse`}
        />
    );
}

function MetricCard({
    label,
    value,
    sub,
    icon: Icon,
    color = "text-white",
}: {
    label: string;
    value: string;
    sub?: string;
    icon: any;
    color?: string;
}) {
    return (
        <div className="glass-card rounded-2xl p-5 hover:-translate-y-0.5 transition-transform">
            <div className="flex items-center gap-2 mb-2">
                <Icon className="w-3.5 h-3.5 text-white/30" />
                <span className="text-[10px] uppercase tracking-[1.5px] text-white/40 font-medium">
                    {label}
                </span>
            </div>
            <div className={`text-3xl font-bold ${color} leading-none`}>{value}</div>
            {sub && <div className="text-[11px] text-white/30 mt-1.5">{sub}</div>}
        </div>
    );
}

function MiniBar({
    label,
    value,
    max,
    unit,
    gradient,
}: {
    label: string;
    value: number;
    max: number;
    unit: string;
    gradient: string;
}) {
    const pct = max > 0 ? (value / max) * 100 : 0;
    return (
        <div>
            <div className="flex justify-between text-[10px] text-white/40 mb-1">
                <span>{label}</span>
                <span>
                    {value}
                    {unit} / {max}
                    {unit}
                </span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                <div
                    className={`h-full rounded-full transition-all duration-700 ${gradient}`}
                    style={{ width: `${pct}%` }}
                />
            </div>
        </div>
    );
}

// ── Simple canvas chart ──
function LossChart({
    history,
}: {
    history: Array<{ train: number; valid: number }>;
}) {
    const canvasRef = useCallback(
        (canvas: HTMLCanvasElement | null) => {
            if (!canvas || history.length < 2) return;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;

            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const w = rect.width;
            const h = rect.height;
            const pad = { top: 20, right: 20, bottom: 30, left: 50 };
            const plotW = w - pad.left - pad.right;
            const plotH = h - pad.top - pad.bottom;

            ctx.clearRect(0, 0, w, h);

            const allVals = history.flatMap((d) => [d.train, d.valid]).filter(Boolean);
            const minY = Math.min(...allVals) * 0.9;
            const maxY = Math.max(...allVals) * 1.1;

            const toX = (i: number) => pad.left + (i / (history.length - 1)) * plotW;
            const toY = (v: number) =>
                pad.top + (1 - (v - minY) / (maxY - minY)) * plotH;

            // Grid
            ctx.strokeStyle = "rgba(255,255,255,0.04)";
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = pad.top + (plotH / 4) * i;
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(w - pad.right, y);
                ctx.stroke();

                ctx.fillStyle = "rgba(255,255,255,0.25)";
                ctx.font = "10px Inter, sans-serif";
                ctx.textAlign = "right";
                const val = maxY - ((maxY - minY) / 4) * i;
                ctx.fillText(val.toFixed(3), pad.left - 8, y + 3);
            }

            // X labels
            ctx.fillStyle = "rgba(255,255,255,0.25)";
            ctx.textAlign = "center";
            const step = Math.max(1, Math.floor(history.length / 5));
            for (let i = 0; i < history.length; i += step) {
                ctx.fillText(`${i + 1}`, toX(i), h - 8);
            }
            ctx.fillText("Epoch", w / 2, h - 0);

            // Draw lines
            const drawLine = (
                data: number[],
                color: string,
                fillColor: string
            ) => {
                if (data.length < 2) return;
                ctx.beginPath();
                ctx.moveTo(toX(0), toY(data[0]));
                for (let i = 1; i < data.length; i++) {
                    ctx.lineTo(toX(i), toY(data[i]));
                }
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();

                // Fill
                ctx.lineTo(toX(data.length - 1), pad.top + plotH);
                ctx.lineTo(toX(0), pad.top + plotH);
                ctx.closePath();
                ctx.fillStyle = fillColor;
                ctx.fill();
            };

            drawLine(
                history.map((h) => h.train),
                "#34d399",
                "rgba(52,211,153,0.08)"
            );
            drawLine(
                history.map((h) => h.valid),
                "#60a5fa",
                "rgba(96,165,250,0.08)"
            );

            // Legend
            const legendX = w - pad.right - 140;
            const legendY = pad.top + 5;
            [
                { label: "Train Loss", color: "#34d399" },
                { label: "Valid Loss", color: "#60a5fa" },
            ].forEach((item, i) => {
                ctx.fillStyle = item.color;
                ctx.fillRect(legendX + i * 80, legendY, 10, 3);
                ctx.fillStyle = "rgba(255,255,255,0.5)";
                ctx.font = "10px Inter, sans-serif";
                ctx.textAlign = "left";
                ctx.fillText(item.label, legendX + 14 + i * 80, legendY + 4);
            });
        },
        [history]
    );

    if (history.length < 2) {
        return (
            <div className="glass-card rounded-2xl p-6">
                <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <BarChart3 className="w-3.5 h-3.5" /> Loss Over Epochs
                </h3>
                <div className="h-[240px] flex items-center justify-center text-white/20 text-sm">
                    Waiting for training to start...
                </div>
            </div>
        );
    }

    return (
        <div className="glass-card rounded-2xl p-6">
            <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest mb-4 flex items-center gap-2">
                <BarChart3 className="w-3.5 h-3.5" /> Loss Over Epochs
            </h3>
            <canvas
                ref={canvasRef}
                className="w-full h-[240px]"
                style={{ width: "100%", height: "240px" }}
            />
        </div>
    );
}

export default function DevDashboard() {
    const [metrics, setMetrics] = useState<Metrics | null>(null);
    const [connected, setConnected] = useState(false);
    const [lastUpdate, setLastUpdate] = useState("");
    const [vmUpHours, setVmUpHours] = useState(0);

    useEffect(() => {
        // VM started at approx 09:46 UTC on Feb 13
        const vmStart = new Date("2026-02-13T09:46:00Z");
        const updateTimer = () => {
            setVmUpHours(
                (Date.now() - vmStart.getTime()) / 3600000
            );
        };
        updateTimer();
        const t = setInterval(updateTimer, 60000);
        return () => clearInterval(t);
    }, []);

    useEffect(() => {
        let active = true;
        const poll = async () => {
            try {
                const res = await fetch(METRICS_URL, { signal: AbortSignal.timeout(8000) });
                const data = await res.json();
                if (active) {
                    setMetrics(data);
                    setConnected(true);
                    setLastUpdate(new Date().toLocaleTimeString());
                }
            } catch {
                if (active) setConnected(false);
            }
        };
        poll();
        const id = setInterval(poll, POLL_INTERVAL);
        return () => {
            active = false;
            clearInterval(id);
        };
    }, []);

    const cost = (vmUpHours * SPOT_PRICE).toFixed(2);
    const pct = metrics
        ? Math.round((metrics.current_epoch / metrics.total_epochs) * 100)
        : 0;

    return (
        <div className="min-h-screen bg-[#08080d]">
            {/* Header */}
            <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#08080d]/80 border-b border-white/5">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link
                            href="/"
                            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                        >
                            <ArrowLeft className="w-4 h-4 text-white/60" />
                        </Link>
                        <div>
                            <h1 className="text-lg font-semibold bg-gradient-to-r from-violet-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent">
                                ⚡ Training Dashboard
                            </h1>
                            <p className="text-[11px] text-white/30 mt-0.5">
                                svoice-trainer • NVIDIA L4 • Spot Instance
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-white/40">
                        <div className="flex items-center gap-1.5">
                            {connected ? (
                                <Wifi className="w-3.5 h-3.5 text-emerald-400" />
                            ) : (
                                <WifiOff className="w-3.5 h-3.5 text-red-400" />
                            )}
                            <StatusDot status={metrics?.status ?? "unknown"} />
                            <span className="font-medium">
                                {connected
                                    ? metrics?.status ?? "connecting..."
                                    : "VM offline"}
                            </span>
                        </div>
                        <span className="text-white/20">|</span>
                        <span>{lastUpdate || "—"}</span>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
                {/* Progress Bar */}
                <div>
                    <div className="h-6 bg-white/[0.03] rounded-xl overflow-hidden border border-white/5">
                        <div
                            className="h-full rounded-xl bg-gradient-to-r from-indigo-500 via-violet-500 to-purple-500 transition-all duration-1000 flex items-center justify-center text-[10px] font-bold text-white min-w-[40px]"
                            style={{ width: `${Math.max(pct, 2)}%` }}
                        >
                            {pct}%
                        </div>
                    </div>
                    <div className="flex justify-between mt-2 text-[11px] text-white/30">
                        <span>
                            Epoch {metrics?.current_epoch ?? 0} / {metrics?.total_epochs ?? 200}
                        </span>
                        <span>
                            {pct > 0 && metrics?.current_epoch
                                ? `~${Math.round(
                                    ((vmUpHours / metrics.current_epoch) *
                                        (200 - metrics.current_epoch)) /
                                    24
                                )} days remaining`
                                : "ETA: waiting for training..."}
                        </span>
                    </div>
                </div>

                {/* Metric Cards */}
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                    <MetricCard
                        label="Epoch"
                        value={metrics?.current_epoch?.toString() ?? "—"}
                        sub="of 200"
                        icon={Activity}
                        color="text-violet-400"
                    />
                    <MetricCard
                        label="Train Loss"
                        value={metrics?.train_loss ? metrics.train_loss.toFixed(4) : "—"}
                        sub="lower is better"
                        icon={TrendingDown}
                        color="text-emerald-400"
                    />
                    <MetricCard
                        label="Valid Loss"
                        value={metrics?.valid_loss ? metrics.valid_loss.toFixed(4) : "—"}
                        sub={`Best: ${metrics?.best_loss && metrics.best_loss < 999 ? metrics.best_loss.toFixed(4) : "—"}`}
                        icon={TrendingDown}
                        color="text-blue-400"
                    />
                    <MetricCard
                        label="Learning Rate"
                        value={metrics?.lr ? metrics.lr.toExponential(1) : "—"}
                        sub="plateau sched"
                        icon={Zap}
                        color="text-cyan-400"
                    />
                    <MetricCard
                        label="Cost"
                        value={`$${cost}`}
                        sub="of $200 budget"
                        icon={DollarSign}
                        color="text-amber-400"
                    />
                    <MetricCard
                        label="Uptime"
                        value={`${vmUpHours.toFixed(1)}h`}
                        sub={`$${SPOT_PRICE}/hr spot`}
                        icon={Clock}
                        color="text-white/60"
                    />
                </div>

                {/* GPU + Chart Row */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    {/* GPU Card */}
                    <div className="glass-card rounded-2xl p-5 space-y-3">
                        <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest flex items-center gap-2">
                            <Cpu className="w-3.5 h-3.5" /> GPU
                        </h3>
                        <div className="text-sm font-medium text-white/70">
                            {metrics?.gpu_name || "—"}
                        </div>
                        <div className="space-y-2.5">
                            <div>
                                <div className="flex justify-between text-[10px] text-white/40 mb-1">
                                    <span className="flex items-center gap-1">
                                        <Thermometer className="w-2.5 h-2.5" /> Temperature
                                    </span>
                                    <span>{metrics?.gpu_temp ?? 0}°C</span>
                                </div>
                                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                    <div
                                        className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-amber-400 to-red-400 transition-all duration-700"
                                        style={{ width: `${metrics?.gpu_temp ?? 0}%` }}
                                    />
                                </div>
                            </div>
                            <MiniBar
                                label="Utilization"
                                value={metrics?.gpu_util ?? 0}
                                max={100}
                                unit="%"
                                gradient="bg-gradient-to-r from-indigo-500 to-violet-500"
                            />
                            <MiniBar
                                label="VRAM"
                                value={metrics?.gpu_mem_used ?? 0}
                                max={metrics?.gpu_mem_total ?? 23034}
                                unit=" MiB"
                                gradient="bg-gradient-to-r from-blue-500 to-cyan-500"
                            />
                        </div>
                        <div className="pt-2 border-t border-white/5">
                            <div className="flex justify-between text-[10px] text-white/30">
                                <span className="flex items-center gap-1">
                                    <HardDrive className="w-2.5 h-2.5" /> Disk
                                </span>
                                <span>
                                    {metrics?.disk_used ?? "—"} / {metrics?.disk_total ?? "—"} (
                                    {metrics?.disk_pct ?? 0}%)
                                </span>
                            </div>
                            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden mt-1">
                                <div
                                    className="h-full rounded-full bg-gradient-to-r from-violet-500 to-purple-500 transition-all duration-700"
                                    style={{ width: `${metrics?.disk_pct ?? 0}%` }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Chart */}
                    <div className="lg:col-span-2">
                        <LossChart history={metrics?.history ?? []} />
                    </div>
                </div>

                {/* Training Configuration */}
                <div className="glass-card rounded-2xl p-5">
                    <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest mb-3">
                        ⚙️ Training Configuration
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                        {[
                            { label: "Encoder (N)", value: "256", paper: "128", better: true },
                            { label: "Hidden (H)", value: "256", paper: "128", better: true },
                            { label: "Layers (R)", value: "10", paper: "6", better: true },
                            { label: "Speakers (C)", value: "10", paper: "2-5", better: true },
                            { label: "Sample Rate", value: "16kHz", paper: "8kHz", better: true },
                            { label: "Batch Size", value: "2", paper: "4", better: false },
                            { label: "Learning Rate", value: "3e-4", paper: "5e-4", better: null },
                            { label: "Epochs", value: "200", paper: "100", better: true },
                            { label: "LR Schedule", value: "plateau", paper: "step", better: true },
                            { label: "Segment", value: "4s", paper: "4s", better: null },
                            { label: "Dataset", value: "20k", paper: "~20k", better: null },
                            { label: "Noise", value: "WHAM!", paper: "WHAM!", better: null },
                        ].map((cfg) => (
                            <div key={cfg.label} className="bg-white/[0.02] border border-white/[0.05] rounded-lg p-2.5 text-center">
                                <div className="text-[10px] text-white/30 mb-1">{cfg.label}</div>
                                <div className="text-sm font-semibold text-white/80">{cfg.value}</div>
                                <div className={`text-[9px] mt-0.5 ${cfg.better === true ? "text-emerald-400/60" :
                                        cfg.better === false ? "text-amber-400/60" :
                                            "text-white/20"
                                    }`}>
                                    paper: {cfg.paper} {cfg.better === true ? "▲" : cfg.better === false ? "▼" : "="}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
                {/* Log */}
                <div className="glass-card rounded-2xl p-5">
                    <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest mb-3">
                        📋 Live Log
                    </h3>
                    <div className="max-h-[200px] overflow-y-auto scrollbar-thin">
                        {(metrics?.log_tail ?? []).map((line, i) => (
                            <div
                                key={i}
                                className={`font-mono text-[11px] leading-relaxed ${line.includes("epoch") || line.includes("loss")
                                    ? "text-violet-400/80"
                                    : line.includes("ERROR") || line.includes("error")
                                        ? "text-red-400/80"
                                        : line.includes("WARNING")
                                            ? "text-amber-400/80"
                                            : "text-white/30"
                                    }`}
                            >
                                {line}
                            </div>
                        ))}
                        {(!metrics?.log_tail || metrics.log_tail.length === 0) && (
                            <div className="text-white/20 text-xs">
                                {connected ? "No logs yet..." : "Connecting to VM..."}
                            </div>
                        )}
                    </div>
                </div>

                {/* Watchdog Alerts */}
                {metrics?.watchdog?.alerts?.length ? (
                    <div className="glass-card rounded-2xl p-5">
                        <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-2">
                                <Shield className={`w-4 h-4 ${metrics?.watchdog?.status === "watching" || metrics?.watchdog?.status?.startsWith("watching")
                                    ? "text-emerald-400"
                                    : metrics?.watchdog?.status === "complete"
                                        ? "text-violet-400"
                                        : "text-white/40"
                                    }`} />
                                <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest">
                                    🛡️ Watchdog
                                </h3>
                            </div>
                            <div className="text-[10px] text-white/20">
                                {metrics?.watchdog?.last_check && `Last check: ${metrics.watchdog.last_check.split(" ")[1] || metrics.watchdog.last_check}`}
                            </div>
                        </div>
                        <div className="space-y-1.5 max-h-[180px] overflow-y-auto scrollbar-thin">
                            {[...(metrics?.watchdog?.alerts ?? [])].reverse().slice(0, 10).map((alert, i) => (
                                <div
                                    key={i}
                                    className={`flex items-start gap-2 p-2 rounded-lg text-[11px] ${alert.level === "critical"
                                        ? "bg-red-500/10 border border-red-500/20"
                                        : alert.level === "warning"
                                            ? "bg-amber-500/10 border border-amber-500/20"
                                            : alert.level === "action"
                                                ? "bg-violet-500/10 border border-violet-500/20"
                                                : "bg-white/[0.02] border border-white/[0.05]"
                                        }`}
                                >
                                    {alert.level === "critical" ? (
                                        <AlertTriangle className="w-3.5 h-3.5 text-red-400 mt-0.5 shrink-0" />
                                    ) : alert.level === "warning" ? (
                                        <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
                                    ) : alert.level === "action" ? (
                                        <Zap className="w-3.5 h-3.5 text-violet-400 mt-0.5 shrink-0" />
                                    ) : (
                                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400/60 mt-0.5 shrink-0" />
                                    )}
                                    <div className="min-w-0">
                                        <div className={`${alert.level === "critical" ? "text-red-300" :
                                            alert.level === "warning" ? "text-amber-300" :
                                                alert.level === "action" ? "text-violet-300" :
                                                    "text-white/50"
                                            }`}>
                                            {alert.message}
                                        </div>
                                        {alert.action_taken && (
                                            <div className="text-[10px] text-white/30 mt-0.5">
                                                → {alert.action_taken}
                                            </div>
                                        )}
                                        <div className="text-[9px] text-white/15 mt-0.5">
                                            {alert.time}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ) : null}

                {/* Download Section */}
                {(metrics?.downloadable_files?.length ?? 0) > 0 && (
                    <div className={`glass-card rounded-2xl p-6 ${metrics?.training_complete
                        ? "ring-1 ring-emerald-500/30 bg-gradient-to-br from-emerald-500/[0.04] to-transparent"
                        : ""
                        }`}>
                        <div className="flex items-center gap-3 mb-4">
                            {metrics?.training_complete ? (
                                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                            ) : (
                                <Download className="w-4 h-4 text-white/40" />
                            )}
                            <div>
                                <h3 className="text-xs font-semibold text-white/40 uppercase tracking-widest">
                                    {metrics?.training_complete
                                        ? "🎉 Training Complete — Download Model"
                                        : "📦 Available Files"}
                                </h3>
                                {metrics?.training_complete && (
                                    <p className="text-[11px] text-emerald-400/60 mt-0.5">
                                        Best loss: {metrics?.best_loss?.toFixed(4)} • {metrics?.total_epochs} epochs
                                    </p>
                                )}
                            </div>
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                            {metrics?.downloadable_files?.map((file) => (
                                <a
                                    key={file.name}
                                    href={`/api/download?file=${file.name}`}
                                    download={file.name}
                                    className="flex items-center justify-between p-4 rounded-xl bg-white/[0.03] hover:bg-white/[0.08] border border-white/[0.06] hover:border-violet-500/30 transition-all group cursor-pointer"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${file.name === "best.th"
                                            ? "bg-emerald-500/20"
                                            : file.name === "checkpoint.th"
                                                ? "bg-violet-500/20"
                                                : "bg-blue-500/20"
                                            }`}>
                                            <Download className={`w-3.5 h-3.5 ${file.name === "best.th"
                                                ? "text-emerald-400"
                                                : file.name === "checkpoint.th"
                                                    ? "text-violet-400"
                                                    : "text-blue-400"
                                                }`} />
                                        </div>
                                        <div>
                                            <div className="text-sm font-medium text-white/80 group-hover:text-white transition-colors">
                                                {file.name}
                                            </div>
                                            <div className="text-[10px] text-white/30">
                                                {file.name === "best.th"
                                                    ? "Best model weights"
                                                    : file.name === "checkpoint.th"
                                                        ? "Latest checkpoint"
                                                        : "Training history"}
                                            </div>
                                        </div>
                                    </div>
                                    <span className="text-[11px] text-white/30 group-hover:text-white/50">
                                        {file.size_human}
                                    </span>
                                </a>
                            ))}
                        </div>
                    </div>
                )}
            </main>

            <style jsx>{`
        .glass-card {
          background: linear-gradient(
            145deg,
            rgba(255, 255, 255, 0.03),
            rgba(255, 255, 255, 0.01)
          );
          border: 1px solid rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(12px);
        }
      `}</style>
        </div>
    );
}
