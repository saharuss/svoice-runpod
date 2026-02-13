#!/usr/bin/env python3
"""
Training Watchdog — monitors SVoice training and takes automatic corrective actions.

Runs alongside training in the screen session. Checks every 10 minutes.

Auto-actions:
  - Restart training if process crashed
  - Reduce LR if loss plateaus for 30+ epochs
  - Clean old checkpoints if disk > 85%
  - Detect NaN/Inf loss → restore last good checkpoint + restart with lower LR

Alerts (written to ~/watchdog_alerts.json for dashboard):
  - 10-speaker training not converging after 50 epochs → suggest fallback to 5
  - Any auto-action taken
"""
import json
import os
import subprocess
import sys
import time
import glob
import shutil
from datetime import datetime

# ── Config ──
CHECK_INTERVAL = 600  # 10 minutes
HISTORY_GLOB = os.path.expanduser("~/svoice_demo/outputs/exp_*/history.json")
CONFIG_GLOB = os.path.expanduser("~/svoice_demo/outputs/exp_*/conf.yml")
CHECKPOINT_GLOB = os.path.expanduser("~/svoice_demo/outputs/exp_*/checkpoint.th")
BEST_GLOB = os.path.expanduser("~/svoice_demo/outputs/exp_*/best.th")
ALERTS_FILE = os.path.expanduser("~/watchdog_alerts.json")
WATCHDOG_LOG = os.path.expanduser("~/watchdog.log")
TRAIN_CMD = ("cd ~/svoice_demo && python train.py "
             "sample_rate=16000 "
             "swave.N=256 swave.H=256 swave.R=10 swave.C=10 "
             "epochs=200 lr=3e-4 "
             "lr_sched=plateau plateau.factor=0.5 plateau.patience=5 "
             "eval_every=5 batch_size=2 num_workers=6 checkpoint=True")

# Thresholds
PLATEAU_EPOCHS = 30       # epochs without improvement before LR reduction
PLATEAU_THRESHOLD = 0.001 # minimum improvement to not be "plateau"
NAN_CHECK = True
DISK_WARN_PCT = 85
CONVERGENCE_CHECK_EPOCH = 50  # after this epoch, check if 10-speaker is working
CONVERGENCE_MIN_IMPROVEMENT = 0.05  # loss must drop at least 5% from epoch 1
LR_REDUCTION_FACTOR = 0.5
MAX_LR_REDUCTIONS = 3

# State
lr_reductions = 0


def log(msg):
    """Write to watchdog log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")


def load_alerts():
    """Load existing alerts."""
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"alerts": [], "auto_actions": [], "status": "watching"}


def save_alerts(data):
    """Save alerts for dashboard."""
    with open(ALERTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_alert(alerts_data, level, message, action_taken=None):
    """Add an alert entry."""
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,  # "info", "warning", "critical", "action"
        "message": message,
    }
    if action_taken:
        entry["action_taken"] = action_taken
        alerts_data["auto_actions"].append(entry)
    alerts_data["alerts"].append(entry)
    # Keep only last 50 alerts
    alerts_data["alerts"] = alerts_data["alerts"][-50:]
    alerts_data["auto_actions"] = alerts_data["auto_actions"][-20:]
    save_alerts(alerts_data)
    log(f"[{level.upper()}] {message}" + (f" → {action_taken}" if action_taken else ""))


def get_history():
    """Load training history."""
    files = glob.glob(HISTORY_GLOB)
    if not files:
        return None
    try:
        with open(files[0]) as f:
            return json.load(f)
    except Exception:
        return None


def get_output_dir():
    """Get training output directory."""
    dirs = glob.glob(os.path.expanduser("~/svoice_demo/outputs/exp_*/"))
    return dirs[0] if dirs else None


def is_training_running():
    """Check if training process is alive."""
    try:
        out = subprocess.check_output(
            "pgrep -f 'python.*train.py'", shell=True, text=True, timeout=5
        ).strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def is_dataset_generating():
    """Check if dataset generation is still running."""
    try:
        out = subprocess.check_output(
            "pgrep -f 'python.*make_dataset'", shell=True, text=True, timeout=5
        ).strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def get_disk_usage_pct():
    """Get root disk usage percentage."""
    try:
        out = subprocess.check_output("df / | tail -1", shell=True, text=True, timeout=5)
        parts = out.split()
        for p in parts:
            if p.endswith('%'):
                return int(p.replace('%', ''))
    except Exception:
        pass
    return 0


def check_nan_loss(history):
    """Check if latest loss is NaN or Inf."""
    if not history:
        return False
    latest = history[-1]
    for key in ["train", "valid"]:
        val = latest.get(key, 0)
        if val is None or (isinstance(val, float) and (val != val or val == float('inf'))):
            return True
        # Also check for absurdly high values (likely diverged)
        if isinstance(val, (int, float)) and abs(val) > 1e6:
            return True
    return False


def check_plateau(history):
    """Check if validation loss has plateaued."""
    if len(history) < PLATEAU_EPOCHS:
        return False
    recent = history[-PLATEAU_EPOCHS:]
    best_recent = min(h.get("valid", 999) for h in recent)
    older = history[:-PLATEAU_EPOCHS]
    if not older:
        return False
    best_older = min(h.get("valid", 999) for h in older)
    # Plateau if no improvement beyond threshold
    improvement = best_older - best_recent
    return improvement < PLATEAU_THRESHOLD


def check_convergence(history):
    """After CONVERGENCE_CHECK_EPOCH, verify model is actually learning."""
    if len(history) < CONVERGENCE_CHECK_EPOCH:
        return True  # too early to tell
    first_loss = history[0].get("valid", 999)
    best_loss = min(h.get("valid", 999) for h in history)
    if first_loss == 0:
        return True
    improvement_pct = (first_loss - best_loss) / abs(first_loss)
    return improvement_pct >= CONVERGENCE_MIN_IMPROVEMENT


def restart_training(alerts_data, reason):
    """Restart the training process."""
    log(f"Restarting training: {reason}")
    add_alert(alerts_data, "action", f"Training restart triggered: {reason}",
              action_taken="Auto-restarted training process")
    try:
        # Kill any zombie training processes
        subprocess.run("pkill -f 'python.*train.py'", shell=True, timeout=10)
        time.sleep(5)
        # Restart training in background
        subprocess.Popen(
            f"nohup {TRAIN_CMD} > ~/train_output.log 2>&1 &",
            shell=True
        )
        log("Training restarted successfully")
    except Exception as e:
        log(f"Failed to restart training: {e}")


def reduce_learning_rate(alerts_data):
    """Reduce LR by modifying the config and restarting."""
    global lr_reductions
    if lr_reductions >= MAX_LR_REDUCTIONS:
        add_alert(alerts_data, "warning",
                  f"Learning rate already reduced {lr_reductions} times, not reducing further",
                  action_taken=None)
        return

    config_files = glob.glob(CONFIG_GLOB)
    if not config_files:
        return

    try:
        import yaml
    except ImportError:
        # If yaml not available, use regex approach
        import re
        config_path = config_files[0]
        with open(config_path) as f:
            content = f.read()
        match = re.search(r'lr:\s*([\d.e-]+)', content)
        if match:
            old_lr = float(match.group(1))
            new_lr = old_lr * LR_REDUCTION_FACTOR
            content = re.sub(r'lr:\s*[\d.e-]+', f'lr: {new_lr}', content)
            with open(config_path, 'w') as f:
                f.write(content)
            lr_reductions += 1
            add_alert(alerts_data, "action",
                      f"Loss plateaued for {PLATEAU_EPOCHS} epochs",
                      action_taken=f"Reduced LR: {old_lr:.6f} → {new_lr:.6f} (reduction #{lr_reductions})")
            restart_training(alerts_data, "LR reduction applied")
        return

    config_path = config_files[0]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    old_lr = config.get("optim", {}).get("lr", config.get("lr", 0.001))
    new_lr = old_lr * LR_REDUCTION_FACTOR

    # Update nested or flat config
    if "optim" in config and "lr" in config["optim"]:
        config["optim"]["lr"] = new_lr
    elif "lr" in config:
        config["lr"] = new_lr

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    lr_reductions += 1
    add_alert(alerts_data, "action",
              f"Loss plateaued for {PLATEAU_EPOCHS} epochs",
              action_taken=f"Reduced LR: {old_lr:.6f} → {new_lr:.6f} (reduction #{lr_reductions})")
    restart_training(alerts_data, "LR reduction applied")


def clean_disk(alerts_data):
    """Remove old checkpoints, keep only best + latest."""
    out_dir = get_output_dir()
    if not out_dir:
        return

    # Find all checkpoint files (epoch-specific ones)
    checkpoint_files = sorted(glob.glob(os.path.join(out_dir, "checkpoint_*.th")))
    if len(checkpoint_files) <= 2:
        return

    # Keep the latest 2, delete the rest
    to_delete = checkpoint_files[:-2]
    freed = 0
    for f in to_delete:
        try:
            size = os.path.getsize(f)
            os.remove(f)
            freed += size
        except Exception:
            pass

    if freed > 0:
        freed_mb = freed / (1024 * 1024)
        add_alert(alerts_data, "info",
                  f"Disk at {get_disk_usage_pct()}% — cleaned {freed_mb:.0f}MB of old checkpoints",
                  action_taken=f"Deleted {len(to_delete)} old checkpoint files")


def run_watchdog():
    """Main watchdog loop."""
    log("=" * 60)
    log("Training Watchdog started")
    log(f"Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL//60}min)")
    log(f"Plateau detection: {PLATEAU_EPOCHS} epochs")
    log(f"Convergence check at epoch: {CONVERGENCE_CHECK_EPOCH}")
    log("=" * 60)

    alerts_data = load_alerts()
    alerts_data["status"] = "watching"
    add_alert(alerts_data, "info", "Watchdog started — monitoring training pipeline")

    convergence_alerted = False
    last_epoch_count = 0

    while True:
        try:
            alerts_data = load_alerts()
            alerts_data["status"] = "watching"
            alerts_data["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ── Check 1: Is dataset gen still running? ──
            if is_dataset_generating():
                alerts_data["status"] = "watching (dataset generating)"
                save_alerts(alerts_data)
                log("Dataset generation still running, waiting...")
                time.sleep(CHECK_INTERVAL)
                continue

            # ── Check 2: Is training running? ──
            if not is_training_running():
                history = get_history()
                if history and len(history) >= 200:
                    # Training completed normally!
                    alerts_data["status"] = "complete"
                    add_alert(alerts_data, "info",
                              f"🎉 Training completed! {len(history)} epochs, "
                              f"best loss: {min(h.get('valid', 999) for h in history):.4f}")
                    save_alerts(alerts_data)
                    log("Training completed normally. Watchdog exiting.")
                    return
                elif history and len(history) > last_epoch_count:
                    # Training was running but crashed
                    last_epoch_count = len(history)
                    add_alert(alerts_data, "warning",
                              f"Training process not found but history shows {len(history)} epochs — possible crash")
                    restart_training(alerts_data, "Process crash detected")
                else:
                    # No history yet and no training — might still be setting up
                    log("No training process found, no history yet. Waiting...")

            # ── Check 3: Load history and analyze ──
            history = get_history()
            if history and len(history) > 0:
                current_epochs = len(history)
                latest = history[-1]
                train_loss = latest.get("train", 0)
                valid_loss = latest.get("valid", 0)
                best_loss = min(h.get("valid", 999) for h in history)

                # Periodic progress log
                if current_epochs > last_epoch_count:
                    log(f"Epoch {current_epochs}: train={train_loss:.4f}, valid={valid_loss:.4f}, best={best_loss:.4f}")
                    last_epoch_count = current_epochs

                # ── Check 3a: NaN/Inf loss ──
                if check_nan_loss(history):
                    add_alert(alerts_data, "critical",
                              f"NaN/Inf loss detected at epoch {current_epochs}!",
                              action_taken="Restoring last checkpoint and reducing LR")
                    # Restore best checkpoint as the current one
                    best_files = glob.glob(BEST_GLOB)
                    ckpt_files = glob.glob(CHECKPOINT_GLOB)
                    if best_files and ckpt_files:
                        try:
                            shutil.copy2(best_files[0], ckpt_files[0])
                            log("Restored best.th as checkpoint.th")
                        except Exception as e:
                            log(f"Failed to restore checkpoint: {e}")
                    reduce_learning_rate(alerts_data)

                # ── Check 3b: Loss plateau ──
                elif check_plateau(history):
                    if lr_reductions < MAX_LR_REDUCTIONS:
                        reduce_learning_rate(alerts_data)
                    else:
                        add_alert(alerts_data, "warning",
                                  f"Loss still plateaued after {lr_reductions} LR reductions. "
                                  "Consider reducing speaker count or increasing dataset.")

                # ── Check 3c: Convergence check ──
                if current_epochs >= CONVERGENCE_CHECK_EPOCH and not convergence_alerted:
                    if not check_convergence(history):
                        convergence_alerted = True
                        first_loss = history[0].get("valid", 999)
                        improvement_pct = ((first_loss - best_loss) / abs(first_loss)) * 100 if first_loss != 0 else 0
                        add_alert(alerts_data, "critical",
                                  f"⚠️ 10-speaker model showing poor convergence after {current_epochs} epochs. "
                                  f"Loss only improved {improvement_pct:.1f}% (need >{CONVERGENCE_MIN_IMPROVEMENT*100}%). "
                                  f"Consider falling back to 5 speakers. "
                                  f"First loss: {first_loss:.4f}, Best: {best_loss:.4f}")

            # ── Check 4: Disk usage ──
            disk_pct = get_disk_usage_pct()
            if disk_pct >= DISK_WARN_PCT:
                clean_disk(alerts_data)

            save_alerts(alerts_data)

        except Exception as e:
            log(f"Watchdog error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run_watchdog()
