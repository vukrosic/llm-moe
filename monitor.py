#!/usr/bin/env python3
import time
import subprocess
import threading
import signal
import sys
from datetime import datetime


class GPUMonitor:
    def __init__(self, interval=2):
        self.interval = interval
        self.running = False
        self.thread = None
        
    def get_gpu_stats(self):
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                stats = []
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        stats.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'utilization': int(parts[2]),
                            'memory_used': int(parts[3]),
                            'memory_total': int(parts[4]),
                            'temperature': int(parts[5])
                        })
                return stats
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        return []
    
    def monitor_loop(self):
        print("🔍 GPU Monitoring Started (Press Ctrl+C to stop)")
        print("=" * 80)
        while self.running:
            stats = self.get_gpu_stats()
            if stats:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] GPU Status:")
                total_util = 0
                for gpu in stats:
                    util = gpu['utilization']
                    mem_used = gpu['memory_used']
                    mem_total = gpu['memory_total']
                    mem_percent = (mem_used / mem_total) * 100
                    temp = gpu['temperature']
                    if util < 20:
                        util_color = "🔴"
                    elif util < 60:
                        util_color = "🟡"
                    else:
                        util_color = "🟢"
                    print(f"  GPU {gpu['index']}: {util_color} {util:3d}% util | {mem_used:5d}/{mem_total:5d}MB ({mem_percent:4.1f}%) | {temp:2d}°C | {gpu['name']}")
                    total_util += util
                if len(stats) > 1:
                    avg_util = total_util / len(stats)
                    max_util = max(g['utilization'] for g in stats)
                    min_util = min(g['utilization'] for g in stats)
                    imbalance = max_util - min_util
                    print(f"  📊 Average: {avg_util:.1f}% | Range: {min_util}%-{max_util}% | Imbalance: {imbalance}%")
                    if imbalance > 30:
                        print(f"  ⚠️  HIGH IMBALANCE DETECTED! ({imbalance}% difference)")
                        print(f"      This suggests uneven workload distribution")
            time.sleep(self.interval)
    
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


def signal_handler(signum, frame):
    print("\n\n🛑 Monitoring stopped by user")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    print("🚀 GPU Utilization Monitor")
    print("Run this alongside your training script to monitor GPU balance")
    try:
        subprocess.run(['nvidia-smi', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ nvidia-smi not found. This tool requires NVIDIA GPUs.")
        sys.exit(1)
    monitor = GPUMonitor(interval=2)
    monitor.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("\n🛑 Monitoring stopped")


if __name__ == "__main__":
    main()
