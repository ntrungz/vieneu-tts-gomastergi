"""Kaggle script kernel: train 2 VieNeu-TTS v3 Turbo LoRA runs in parallel,
one per T4 (train_lora.py is single-GPU only). Run A: r16 backbone (default).
Run B: r32 + acoustic decoder. Both --merge. Outputs land in ./VieNeu-TTS/finetune/output/."""
import os, subprocess, sys, time

def run(cmd, env=None, log="train.log"):
    print(f"$ {' '.join(cmd)}", flush=True)
    e = dict(os.environ); e.update(env or {})
    with open(log, "ab") as lf:
        p = subprocess.Popen(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lf.write(line.encode()); lf.flush()
        p.wait()
    print(f"exit={p.returncode} log={log}", flush=True)
    return p.returncode

print("GPU check:", flush=True)
run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], log="gpu.log")
run(["git", "clone", "--depth", "1", "https://github.com/pnnbao97/VieNeu-TTS.git"], log="clone.log")
run([sys.executable, "-m", "pip", "install", "-q", "peft==0.15.2", "accelerate", "datasets", "transformers==4.57.6"], log="pip.log")
# Kaggle image ships torchao 0.10.0; newer peft refuses it. peft 0.15.2 is safe,
# and torchao is unused by this model, so remove it entirely to avoid any check.
run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], log="pip-uninstall-torchao.log")

print("kaggle input contents:", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "NO /kaggle/input", flush=True)
DATA = "/kaggle/input/vieneu-gomastergi-50min/train.parquet"
if not os.path.exists(DATA):
    print("dataset not attached via input; trying kaggle CLI download...", flush=True)
    run(["kaggle", "datasets", "download", "ngnhattrung/vieneu-gomastergi-50min", "-p", ".", "--unzip"], log="ds-download.log")
    DATA = "./train.parquet"
assert os.path.exists(DATA), f"dataset missing: {DATA}"
print("dataset:", os.path.getsize(DATA), "bytes", flush=True)

import threading
results = {}
def train_one(name, extra, gpu, log):
    cmd = [sys.executable, "VieNeu-TTS/finetune/train_lora.py",
           "--data", DATA, "--run", name, "--merge"] + extra
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    rc = run(cmd, env=env, log=log)
    results[name] = rc

t0 = time.time()
threads = [
    threading.Thread(target=train_one, args=("gomastergi-r16", ["--r","16","--alpha","32","--target","backbone","--epochs","3"], 0, "train-r16.log")),
    threading.Thread(target=train_one, args=("gomastergi-r32-all", ["--r","32","--alpha","64","--target","all","--epochs","3"], 1, "train-r32-all.log")),
]
for t in threads: t.start()
for t in threads: t.join()
print(f"ALL DONE in {(time.time()-t0)/60:.1f} min, results={results}", flush=True)
failed = [n for n, rc in results.items() if rc != 0]
if failed:
    raise SystemExit(f"TRAINING FAILED for: {failed} (see train-*.log)")
run(["find", "VieNeu-TTS/finetune/output", "-maxdepth", "3", "-type", "d"], log="find.log")
