#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Stage Monitor (watch_stage.py)

Monitors SLURM job progress by tracking COMPLETION SENTINEL FILES.
Used by unified_pipeline.sh to wait for stages to complete.

V2 UPDATE (April 2026):
- Now uses .done SENTINEL FILES instead of output files to detect completion
- This fixes a critical race condition where output files were created
  immediately when jobs started (via `open(file, 'w')`), causing the
  monitor to report success before processing was actually complete
- Sentinel files are only written AFTER all processing and file writing
  is complete, guaranteeing reliable completion detection

Features:
- Counts .done sentinel files matching a pattern
- Monitors SLURM job status as fallback
- Displays progress with ETA
- Handles both interactive and non-interactive modes
"""

import os
import sys
import time
import glob
import argparse
import subprocess


def is_job_running(job_id):
    """
    Checks if the job (or any of its array tasks) is still in the SLURM queue.
    For array jobs, this returns True if ANY array task is still running/pending.
    """
    try:
        result = subprocess.run(
            ['squeue', '-h', '-j', str(job_id)], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=30
        )
        return len(result.stdout.strip()) > 0
    except subprocess.TimeoutExpired:
        print(f"Warning: squeue timed out for job {job_id}")
        return True
    except Exception as e:
        print(f"Warning: Error checking job status: {e}")
        return True


def count_files(pattern):
    """
    Count files matching pattern(s). 
    Supports space-separated multiple patterns.
    """
    patterns = pattern.split()
    total = 0
    for p in patterns:
        total += len(glob.glob(p))
    return total


def format_time(seconds):
    """Formats seconds to HH:MM:SS"""
    if seconds is None or seconds < 0:
        return "--:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Monitor SLURM job progress by tracking .done sentinel files."
    )
    parser.add_argument("--job_id", required=True, 
                        help="SLURM Job ID to watch")
    parser.add_argument("--pattern", required=True, 
                        help="Sentinel file pattern(s) to count, e.g., 'results_*/job_*.done'")
    parser.add_argument("--total", type=int, required=True, 
                        help="Expected number of sentinel files (usually = number of jobs)")
    parser.add_argument("--name", default="Stage", 
                        help="Name of the stage for display")
    parser.add_argument("--poll_interval", type=int, default=10,
                        help="Seconds between status checks (default: 10)")
    args = parser.parse_args()

    # Detect if interactive terminal
    interactive = sys.stdout.isatty()
    
    current = 0
    start_time = time.time()
    last_update_time = start_time
    
    print(f"\n{'='*60}")
    print(f"  Monitoring: {args.name}")
    print(f"  Job ID: {args.job_id}")
    print(f"  Expected completions: {args.total}")
    print(f"  Sentinel pattern: {args.pattern}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    while True:
        # 1. Update file count
        new_count = count_files(args.pattern)
        current = new_count
        elapsed = time.time() - start_time
        
        # 2. Calculate progress metrics
        if args.total > 0:
            percent = (current / args.total) * 100.0
        else:
            percent = 0
        
        # Calculate ETA
        if current > 0 and elapsed > 0:
            rate = current / elapsed
            remaining = args.total - current
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = format_time(eta_seconds)
        else:
            eta_str = "--:--:--"
        
        # 3. Check success condition (all sentinel files present)
        if current >= args.total:
            print(f"\n[SUCCESS] {args.name}: All {args.total} jobs completed")
            print(f"  (All .done sentinel files present)")
            print(f"  Elapsed time: {format_time(elapsed)}")
            return 0

        # 4. Check SLURM job status as fallback
        if not is_job_running(args.job_id):
            # Job finished - wait a moment for final sentinel file writes
            time.sleep(10)  # Increased from 5s to allow for filesystem sync
            final_count = count_files(args.pattern)
            
            if final_count >= args.total:
                print(f"\n[SUCCESS] {args.name}: All jobs completed")
                print(f"  Elapsed time: {format_time(elapsed)}")
                return 0
            else:
                print(f"\n[WARNING] {args.name}: SLURM job {args.job_id} finished")
                print(f"  But only {final_count}/{args.total} sentinel files found!")
                print(f"  Sentinel pattern: {args.pattern}")
                print(f"  Some jobs may have FAILED - check SLURM logs (.err files)")
                print(f"  Pipeline will continue, but results may be incomplete.")
                # Return non-zero to signal partial completion
                return 1

        # 5. Display progress update
        time_since_update = time.time() - last_update_time
        if time_since_update >= args.poll_interval:
            print(f"[{args.name}] {current}/{args.total} jobs done ({percent:.1f}%) | "
                  f"Elapsed: {format_time(elapsed)} | ETA: {eta_str}")
            sys.stdout.flush()
            last_update_time = time.time()

        # 6. Sleep before next check
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
