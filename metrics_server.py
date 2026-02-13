#!/usr/bin/env python3
"""
Simple HTTP metrics server for the SVoice training VM.
Exposes training metrics as JSON on port 8080.
Also serves model file downloads when training is complete.
Accessed by the Next.js dev dashboard.
"""
import json
import os
import glob
import http.server
import subprocess

PORT = 8080

# Files available for download after training
DOWNLOADABLE_FILES = ["best.th", "checkpoint.th", "history.json"]

def get_output_dir():
    """Find the training output directory."""
    dirs = glob.glob(os.path.expanduser("~/svoice_demo/outputs/exp_*/"))
    return dirs[0] if dirs else None

def get_downloadable_files():
    """Check which model files exist and return their info."""
    out_dir = get_output_dir()
    if not out_dir:
        return []
    files = []
    for name in DOWNLOADABLE_FILES:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            size_bytes = os.path.getsize(path)
            if size_bytes > 0:
                files.append({
                    "name": name,
                    "size_bytes": size_bytes,
                    "size_human": f"{size_bytes / (1024*1024):.1f} MB" if size_bytes > 1024*1024 else f"{size_bytes / 1024:.1f} KB",
                })
    return files

def get_metrics():
    """Collect all training metrics."""
    metrics = {
        "status": "unknown",
        "gpu_name": "",
        "gpu_temp": 0,
        "gpu_util": 0,
        "gpu_mem_used": 0,
        "gpu_mem_total": 0,
        "current_epoch": 0,
        "total_epochs": 200,
        "train_loss": 0,
        "valid_loss": 0,
        "best_loss": 999,
        "lr": 0,
        "disk_used": "",
        "disk_total": "",
        "disk_pct": 0,
        "history": [],
        "log_tail": [],
        "dataset_progress": "",
        "training_complete": False,
        "downloadable_files": [],
    }

    # GPU info
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits",
            shell=True, text=True, timeout=5
        ).strip()
        parts = out.split(", ")
        if len(parts) >= 5:
            metrics["gpu_name"] = parts[0]
            metrics["gpu_temp"] = int(parts[1])
            metrics["gpu_util"] = int(parts[2])
            metrics["gpu_mem_used"] = int(parts[3])
            metrics["gpu_mem_total"] = int(parts[4])
    except Exception:
        pass

    # Disk usage
    try:
        out = subprocess.check_output("df -h / | tail -1", shell=True, text=True, timeout=5).strip()
        parts = out.split()
        if len(parts) >= 5:
            metrics["disk_total"] = parts[1]
            metrics["disk_used"] = parts[2]
            metrics["disk_pct"] = int(parts[4].replace('%', ''))
    except Exception:
        pass

    # Training history
    try:
        history_files = glob.glob(os.path.expanduser("~/svoice_demo/outputs/exp_*/history.json"))
        if history_files:
            with open(history_files[0]) as f:
                history = json.load(f)
                metrics["history"] = history
                if history:
                    latest = history[-1]
                    metrics["current_epoch"] = len(history)
                    metrics["train_loss"] = latest.get("train", 0)
                    metrics["valid_loss"] = latest.get("valid", 0)
                    metrics["best_loss"] = min(h.get("valid", 999) for h in history)
                    metrics["lr"] = latest.get("lr", 0)
                    metrics["status"] = "training"
                    # Check if training is complete
                    if len(history) >= metrics["total_epochs"]:
                        metrics["status"] = "complete"
                        metrics["training_complete"] = True
    except Exception:
        pass

    # Downloadable files
    metrics["downloadable_files"] = get_downloadable_files()

    # Trainer log tail
    try:
        log_files = glob.glob(os.path.expanduser("~/svoice_demo/outputs/exp_*/trainer.log"))
        if log_files:
            with open(log_files[0]) as f:
                lines = f.readlines()
                metrics["log_tail"] = [l.strip() for l in lines[-20:]]
    except Exception:
        pass

    # Dataset generation progress (if no training yet)
    if metrics["current_epoch"] == 0:
        try:
            log_path = os.path.expanduser("~/dataset_gen.log")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    lines = f.readlines()
                    tail = [l.strip() for l in lines[-5:] if l.strip()]
                    metrics["log_tail"] = tail
                    # Parse percentage from wget-style output
                    for line in reversed(tail):
                        parts = line.split()
                        for p in parts:
                            if p.endswith('%') and p[:-1].isdigit():
                                metrics["dataset_progress"] = p
                                metrics["status"] = f"downloading dataset ({p})"
                                break
                        if metrics["dataset_progress"]:
                            break
                    if not metrics["dataset_progress"]:
                        metrics["status"] = "generating dataset"
            else:
                setup_path = os.path.expanduser("~/setup.log")
                if os.path.exists(setup_path):
                    with open(setup_path) as f:
                        lines = f.readlines()
                        metrics["log_tail"] = [l.strip() for l in lines[-5:]]
                        metrics["status"] = "setting up"
                else:
                    metrics["status"] = "waiting"
        except Exception:
            metrics["status"] = "unknown"

    return metrics


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/metrics" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(get_metrics()).encode())
        elif self.path.startswith("/download/"):
            filename = self.path.split("/download/")[-1]
            if filename not in DOWNLOADABLE_FILES:
                self.send_error(400, "Invalid file")
                return
            out_dir = get_output_dir()
            if not out_dir:
                self.send_error(404, "No training output found")
                return
            filepath = os.path.join(out_dir, filename)
            if not os.path.exists(filepath):
                self.send_error(404, f"{filename} not found")
                return
            file_size = os.path.getsize(filepath)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(file_size))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print(f"Metrics server running on port {PORT}")
    server = http.server.HTTPServer(("0.0.0.0", PORT), MetricsHandler)
    server.serve_forever()
