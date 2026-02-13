#!/usr/bin/env python3
"""
SVoice Training Dashboard — Live monitoring for GCP VM training.

Run: python3 training_dashboard.py
Open: http://localhost:8787
"""

import json
import os
import re
import subprocess
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime, timedelta

# ── Configuration ──
VM_NAME = "svoice-trainer"
ZONE = "us-central1-a"
PROJECT = "project-a591a480-81fe-4018-ae8"
POLL_INTERVAL = 30  # seconds between VM polls
PORT = 8787
GCLOUD_PATH = "/opt/homebrew/share/google-cloud-sdk/bin/gcloud"
SPOT_PRICE_PER_HOUR = 0.22  # L4 Spot $/hr

# ── Shared state ──
state = {
    "status": "initializing",
    "vm_status": "unknown",
    "gpu_name": "",
    "gpu_temp": 0,
    "gpu_util": 0,
    "gpu_mem_used": 0,
    "gpu_mem_total": 0,
    "current_epoch": 0,
    "total_epochs": 200,
    "train_loss": 0.0,
    "valid_loss": 0.0,
    "best_loss": 999.0,
    "lr": 0.0,
    "elapsed_hours": 0.0,
    "cost_so_far": 0.0,
    "estimated_total_cost": 0.0,
    "eta": "",
    "history": [],
    "log_tail": [],
    "setup_log": "",
    "last_update": "",
    "error": "",
    "start_time": None,
}
state_lock = threading.Lock()


def run_ssh_command(cmd, timeout=15):
    """Run a command on the VM via gcloud SSH."""
    full_cmd = [
        GCLOUD_PATH, "compute", "ssh", VM_NAME,
        f"--zone={ZONE}", f"--project={PROJECT}",
        f"--command={cmd}",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PATH": f"/opt/homebrew/share/google-cloud-sdk/bin:{os.environ.get('PATH', '')}"},
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        return f"ERROR: {e}"


def parse_gpu_info(raw):
    """Parse nvidia-smi output."""
    info = {"name": "", "temp": 0, "util": 0, "mem_used": 0, "mem_total": 0}
    if not raw or "ERROR" in raw:
        return info
    try:
        parts = raw.strip().split(", ")
        if len(parts) >= 5:
            info["name"] = parts[0]
            info["temp"] = int(parts[1])
            info["util"] = int(parts[2])
            info["mem_used"] = int(parts[3])
            info["mem_total"] = int(parts[4])
    except (ValueError, IndexError):
        pass
    return info


def parse_training_log(raw):
    """Parse history.json from training output."""
    if not raw or "ERROR" in raw or "No such file" in raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def parse_trainer_log_tail(raw):
    """Parse the last N lines of trainer.log."""
    if not raw or "ERROR" in raw or "No such file" in raw:
        return []
    return raw.strip().split("\n")[-20:]


def poll_vm():
    """Background thread that polls the VM for metrics."""
    global state
    
    while True:
        try:
            # ── Check VM status ──
            vm_check = subprocess.run(
                [GCLOUD_PATH, "compute", "instances", "describe", VM_NAME,
                 f"--zone={ZONE}", f"--project={PROJECT}",
                 "--format=value(status)"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "PATH": f"/opt/homebrew/share/google-cloud-sdk/bin:{os.environ.get('PATH', '')}"},
            )
            vm_status = vm_check.stdout.strip()

            with state_lock:
                state["vm_status"] = vm_status
                state["last_update"] = datetime.now().strftime("%H:%M:%S")

            if vm_status != "RUNNING":
                with state_lock:
                    state["status"] = f"VM is {vm_status}"
                    state["error"] = "VM is not running. Restart it to resume training."
                time.sleep(POLL_INTERVAL)
                continue

            # ── GPU info ──
            gpu_raw = run_ssh_command(
                "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total "
                "--format=csv,noheader,nounits 2>/dev/null || echo 'N/A, 0, 0, 0, 0'"
            )
            gpu = parse_gpu_info(gpu_raw)

            # ── Training history ──
            history_raw = run_ssh_command(
                "cat ~/svoice_demo/outputs/exp_*/history.json 2>/dev/null || echo '[]'"
            )
            history = parse_training_log(history_raw)

            # ── Trainer log tail ──
            log_raw = run_ssh_command(
                "tail -30 ~/svoice_demo/outputs/exp_*/trainer.log 2>/dev/null || echo 'No training log yet'"
            )
            log_lines = parse_trainer_log_tail(log_raw)

            # ── Setup log (if no training yet) ──
            setup_raw = ""
            if not history:
                setup_raw = run_ssh_command(
                    "echo '=== SETUP ===' && tail -5 ~/setup.log 2>/dev/null; "
                    "echo '' && echo '=== DATASET GEN ===' && tail -10 ~/dataset_gen.log 2>/dev/null; "
                    "echo '' && echo '=== DISK ===' && df -h / 2>/dev/null | tail -1"
                )

            # ── Parse current metrics ──
            current_epoch = 0
            train_loss = 0.0
            valid_loss = 0.0
            best_loss = 999.0
            lr = 0.0

            if history:
                latest = history[-1] if history else {}
                current_epoch = len(history)
                train_loss = latest.get("train", 0.0)
                valid_loss = latest.get("valid", 0.0)
                best_loss = min(h.get("valid", 999) for h in history) if history else 999
                lr = latest.get("lr", 0.0)

            # ── Calculate timing & cost ──
            with state_lock:
                if state["start_time"] is None and current_epoch > 0:
                    state["start_time"] = datetime.now()

                elapsed_hours = 0.0
                if state["start_time"]:
                    elapsed_hours = (datetime.now() - state["start_time"]).total_seconds() / 3600

                cost_so_far = elapsed_hours * SPOT_PRICE_PER_HOUR
                
                eta = ""
                est_total_cost = 0.0
                if current_epoch > 0 and elapsed_hours > 0:
                    hours_per_epoch = elapsed_hours / current_epoch
                    remaining_epochs = 200 - current_epoch
                    remaining_hours = remaining_epochs * hours_per_epoch
                    eta_dt = datetime.now() + timedelta(hours=remaining_hours)
                    eta = eta_dt.strftime("%b %d, %I:%M %p")
                    est_total_cost = (elapsed_hours + remaining_hours) * SPOT_PRICE_PER_HOUR

                status = "training" if current_epoch > 0 else ("setting up" if setup_raw else "waiting")

                state.update({
                    "status": status,
                    "gpu_name": gpu["name"],
                    "gpu_temp": gpu["temp"],
                    "gpu_util": gpu["util"],
                    "gpu_mem_used": gpu["mem_used"],
                    "gpu_mem_total": gpu["mem_total"],
                    "current_epoch": current_epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "best_loss": best_loss,
                    "lr": lr,
                    "elapsed_hours": round(elapsed_hours, 2),
                    "cost_so_far": round(cost_so_far, 2),
                    "estimated_total_cost": round(est_total_cost, 2),
                    "eta": eta,
                    "history": history,
                    "log_tail": log_lines,
                    "setup_log": setup_raw,
                    "error": "",
                })

        except Exception as e:
            with state_lock:
                state["error"] = str(e)
                state["last_update"] = datetime.now().strftime("%H:%M:%S")

        time.sleep(POLL_INTERVAL)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SVoice Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Inter', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
}
.header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 24px 32px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header h1 {
    font-size: 22px;
    font-weight: 600;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header .meta {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 13px;
    color: #888;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-dot.running { background: #34d399; box-shadow: 0 0 8px #34d39966; animation: pulse 2s infinite; }
.status-dot.stopped { background: #f87171; }
.status-dot.setup { background: #fbbf24; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    padding: 24px 32px;
}
.card {
    background: linear-gradient(145deg, #13131f, #1a1a2e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    transition: transform 0.2s;
}
.card:hover { transform: translateY(-2px); }
.card .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #666;
    margin-bottom: 8px;
}
.card .value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1.1;
}
.card .sub {
    font-size: 12px;
    color: #555;
    margin-top: 4px;
}
.value.purple { color: #a78bfa; }
.value.green { color: #34d399; }
.value.blue { color: #60a5fa; }
.value.amber { color: #fbbf24; }
.value.red { color: #f87171; }
.value.cyan { color: #22d3ee; }

.chart-section {
    padding: 0 32px 24px;
}
.chart-card {
    background: linear-gradient(145deg, #13131f, #1a1a2e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
}
.chart-card h3 {
    font-size: 14px;
    font-weight: 600;
    color: #888;
    margin-bottom: 16px;
}
.chart-container {
    position: relative;
    height: 300px;
}

.progress-section {
    padding: 0 32px 24px;
}
.progress-bar-bg {
    background: #1a1a2e;
    border-radius: 12px;
    height: 24px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.progress-bar-fill {
    height: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #6366f1, #a78bfa, #c084fc);
    transition: width 1s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: white;
    min-width: 40px;
}

.log-section {
    padding: 0 32px 32px;
}
.log-card {
    background: #0d0d14;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    max-height: 300px;
    overflow-y: auto;
}
.log-card h3 {
    font-size: 14px;
    font-weight: 600;
    color: #888;
    margin-bottom: 12px;
}
.log-line {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    color: #888;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-all;
}
.log-line.highlight { color: #a78bfa; }
.log-line.warn { color: #fbbf24; }
.log-line.err { color: #f87171; }

.gpu-bar {
    height: 6px;
    background: #1a1a2e;
    border-radius: 3px;
    margin-top: 8px;
    overflow: hidden;
}
.gpu-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1s ease;
}
.gpu-bar-fill.temp { background: linear-gradient(90deg, #34d399, #fbbf24, #f87171); }
.gpu-bar-fill.util { background: linear-gradient(90deg, #6366f1, #a78bfa); }
.gpu-bar-fill.mem { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
</style>
</head>
<body>

<div class="header">
    <h1>⚡ SVoice Training Dashboard</h1>
    <div class="meta">
        <span id="vm-status"><span class="status-dot"></span>Connecting...</span>
        <span id="last-update">—</span>
    </div>
</div>

<div class="progress-section" style="padding-top: 24px;">
    <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="epoch-progress" style="width: 0%">0%</div>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: #555;">
        <span id="epoch-text">Epoch 0 / 200</span>
        <span id="eta-text">ETA: calculating...</span>
    </div>
</div>

<div class="grid">
    <div class="card">
        <div class="label">Current Epoch</div>
        <div class="value purple" id="current-epoch">—</div>
        <div class="sub" id="epoch-sub">of 200 total</div>
    </div>
    <div class="card">
        <div class="label">Train Loss</div>
        <div class="value green" id="train-loss">—</div>
        <div class="sub">Lower is better</div>
    </div>
    <div class="card">
        <div class="label">Valid Loss</div>
        <div class="value blue" id="valid-loss">—</div>
        <div class="sub" id="best-loss-sub">Best: —</div>
    </div>
    <div class="card">
        <div class="label">Learning Rate</div>
        <div class="value cyan" id="learning-rate">—</div>
        <div class="sub">Plateau scheduler</div>
    </div>
    <div class="card">
        <div class="label">Cost So Far</div>
        <div class="value amber" id="cost">$0.00</div>
        <div class="sub" id="cost-sub">Est. total: —</div>
    </div>
    <div class="card">
        <div class="label">GPU</div>
        <div class="value" id="gpu-name" style="font-size: 18px; color: #888;">—</div>
        <div style="margin-top: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">
                <span>Temp</span><span id="gpu-temp">—°C</span>
            </div>
            <div class="gpu-bar"><div class="gpu-bar-fill temp" id="gpu-temp-bar" style="width: 0%"></div></div>
        </div>
        <div style="margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">
                <span>Utilization</span><span id="gpu-util">—%</span>
            </div>
            <div class="gpu-bar"><div class="gpu-bar-fill util" id="gpu-util-bar" style="width: 0%"></div></div>
        </div>
        <div style="margin-top: 6px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">
                <span>VRAM</span><span id="gpu-mem">— / — MiB</span>
            </div>
            <div class="gpu-bar"><div class="gpu-bar-fill mem" id="gpu-mem-bar" style="width: 0%"></div></div>
        </div>
    </div>
</div>

<div class="chart-section">
    <div class="chart-card">
        <h3>Loss Over Epochs</h3>
        <div class="chart-container">
            <canvas id="lossChart"></canvas>
        </div>
    </div>
</div>

<div class="log-section">
    <div class="log-card">
        <h3>📋 Live Log</h3>
        <div id="log-output"></div>
    </div>
</div>

<script>
const ctx = document.getElementById('lossChart').getContext('2d');
const lossChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Train Loss',
                data: [],
                borderColor: '#34d399',
                backgroundColor: 'rgba(52,211,153,0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 4,
            },
            {
                label: 'Valid Loss',
                data: [],
                borderColor: '#60a5fa',
                backgroundColor: 'rgba(96,165,250,0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 4,
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
            x: {
                title: { display: true, text: 'Epoch', color: '#555' },
                ticks: { color: '#555', maxTicksLimit: 20 },
                grid: { color: 'rgba(255,255,255,0.03)' },
            },
            y: {
                title: { display: true, text: 'Loss', color: '#555' },
                ticks: { color: '#555' },
                grid: { color: 'rgba(255,255,255,0.03)' },
            }
        },
        plugins: {
            legend: { labels: { color: '#888', usePointStyle: true } },
        }
    }
});

async function fetchState() {
    try {
        const res = await fetch('/api/state');
        const data = await res.json();
        updateUI(data);
    } catch (e) {
        console.error('Fetch error:', e);
    }
}

function updateUI(d) {
    // Status
    const statusEl = document.getElementById('vm-status');
    const dotClass = d.vm_status === 'RUNNING' ? (d.status === 'training' ? 'running' : 'setup') : 'stopped';
    statusEl.innerHTML = `<span class="status-dot ${dotClass}"></span>${d.vm_status} — ${d.status}`;
    document.getElementById('last-update').textContent = `Updated: ${d.last_update}`;

    // Progress
    const pct = Math.round((d.current_epoch / d.total_epochs) * 100);
    document.getElementById('epoch-progress').style.width = `${Math.max(pct, 2)}%`;
    document.getElementById('epoch-progress').textContent = `${pct}%`;
    document.getElementById('epoch-text').textContent = `Epoch ${d.current_epoch} / ${d.total_epochs}`;
    document.getElementById('eta-text').textContent = d.eta ? `ETA: ${d.eta}` : 'ETA: calculating...';

    // Cards
    document.getElementById('current-epoch').textContent = d.current_epoch || '—';
    document.getElementById('train-loss').textContent = d.train_loss ? d.train_loss.toFixed(4) : '—';
    document.getElementById('valid-loss').textContent = d.valid_loss ? d.valid_loss.toFixed(4) : '—';
    document.getElementById('best-loss-sub').textContent = `Best: ${d.best_loss < 999 ? d.best_loss.toFixed(4) : '—'}`;
    document.getElementById('learning-rate').textContent = d.lr ? d.lr.toExponential(2) : '—';
    document.getElementById('cost').textContent = `$${d.cost_so_far.toFixed(2)}`;
    document.getElementById('cost-sub').textContent = `Est. total: $${d.estimated_total_cost > 0 ? d.estimated_total_cost.toFixed(2) : '—'} / $200 budget`;

    // GPU
    document.getElementById('gpu-name').textContent = d.gpu_name || '—';
    document.getElementById('gpu-temp').textContent = `${d.gpu_temp}°C`;
    document.getElementById('gpu-temp-bar').style.width = `${d.gpu_temp}%`;
    document.getElementById('gpu-util').textContent = `${d.gpu_util}%`;
    document.getElementById('gpu-util-bar').style.width = `${d.gpu_util}%`;
    document.getElementById('gpu-mem').textContent = `${d.gpu_mem_used} / ${d.gpu_mem_total} MiB`;
    const memPct = d.gpu_mem_total > 0 ? (d.gpu_mem_used / d.gpu_mem_total * 100) : 0;
    document.getElementById('gpu-mem-bar').style.width = `${memPct}%`;

    // Chart
    if (d.history && d.history.length > 0) {
        lossChart.data.labels = d.history.map((_, i) => i + 1);
        lossChart.data.datasets[0].data = d.history.map(h => h.train);
        lossChart.data.datasets[1].data = d.history.map(h => h.valid);
        lossChart.update('none');
    }

    // Log
    const logEl = document.getElementById('log-output');
    const lines = d.status === 'training' ? d.log_tail : (d.setup_log ? d.setup_log.split('\\n') : d.log_tail);
    logEl.innerHTML = (lines || []).map(l => {
        let cls = 'log-line';
        if (l.includes('epoch') || l.includes('Epoch') || l.includes('loss')) cls += ' highlight';
        else if (l.includes('WARNING') || l.includes('warn')) cls += ' warn';
        else if (l.includes('ERROR') || l.includes('error')) cls += ' err';
        return `<div class="${cls}">${escapeHtml(l)}</div>`;
    }).join('');
    logEl.scrollTop = logEl.scrollHeight;
}

function escapeHtml(s) {
    const el = document.createElement('span');
    el.textContent = s;
    return el.innerHTML;
}

// Poll every 10 seconds
fetchState();
setInterval(fetchState, 10000);
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with state_lock:
                # Make a copy, exclude non-serializable
                out = {k: v for k, v in state.items() if k != "start_time"}
            self.wfile.write(json.dumps(out).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    print("═" * 56)
    print("  ⚡ SVoice Training Dashboard")
    print("═" * 56)
    print()
    print(f"  VM:     {VM_NAME}")
    print(f"  Zone:   {ZONE}")
    print(f"  Poll:   every {POLL_INTERVAL}s")
    print()
    print(f"  Dashboard: http://localhost:{PORT}")
    print()
    print("  Press Ctrl+C to stop")
    print("═" * 56)

    # Start background polling
    poller = threading.Thread(target=poll_vm, daemon=True)
    poller.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
