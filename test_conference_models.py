import json
import os
import subprocess
import time

def run_model_battery():
    map_path = "configs/env_map_conference.json"
    with open(map_path, 'r') as f:
        registry = json.load(f)
        
    input_img = os.path.abspath("satimg.jpeg")
    output_dir = "outputs/battery_test"
    os.makedirs(output_dir, exist_ok=True)
    
    report = []
    
    print(f"🚀 Starting Mass Model Test Battery for {len(registry)} models...\n")
    
    for model_name, config in registry.items():
        print(f"[*] Testing: {model_name}...")
        out_img = os.path.join(output_dir, f"{model_name}_test_out.png")
        
        # Build dynamic command
        in_flag = config.get("in_flag", "--input")
        out_flag = config.get("out_flag", "--output")
        weight_flag = config.get("weight_flag", "--weights")
        
        cmd = [config["python_bin"], config["script_path"], in_flag, input_img, out_flag, out_img]
        
        if config.get("weights"):
            cmd.extend([weight_flag, config["weights"]])
        if "extra_args" in config:
            cmd.extend(config["extra_args"])
            
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1" # Forcing it to your specific GPU
        
        start_time = time.time()
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        exec_time = round(time.time() - start_time, 2)
        
        # --- STRICT DIAGNOSTICS ---
        status = "✅ PASS"
        err_detail = ""
        
        if res.returncode != 0:
            status = "❌ FAIL (Crash)"
            # Grab the last 3 lines of the traceback
            err_lines = [line.strip() for line in res.stderr.strip().split('\n') if line.strip()]
            err_detail = " | ".join(err_lines[-3:]) if err_lines else "Unknown Crash"
            
        elif not os.path.exists(out_img):
            status = "❌ FAIL (Lied)"
            err_detail = "Script returned Success (0), but output file is missing."
            
        elif os.path.getsize(out_img) < 1024:
            status = "❌ FAIL (Corrupt)"
            err_detail = f"File is abnormally small: {os.path.getsize(out_img)} bytes."
            
        # Log result
        report_line = f"{status} | {model_name.ljust(15)} | {exec_time}s"
        report.append(report_line)
        if status != "✅ PASS":
            report.append(f"    └─ Reason: {err_detail}")
            
        print(f"    -> {status}\n")

    # Print Final Report
    print("="*60)
    print("📋 FINAL BATTERY REPORT")
    print("="*60)
    for line in report:
        print(line)

if __name__ == "__main__":
    run_model_battery()