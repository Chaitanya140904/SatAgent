import subprocess
import json
import os
import sys
from pathlib import Path

class ConferenceRouter:
    def __init__(self, env_map_path="configs/env_map_conference.json"):
        # SAVE it to self so the whole class can use it
        self.base_dir = Path(__file__).resolve().parents[1]
        map_file = self.base_dir / env_map_path
        
        with open(map_file, 'r') as f:
            self.registry = json.load(f)

    def execute_expert(self, model_key: str, input_path: str, output_path: str, gpu_id: int, mask_path: str = None) -> bool:
        if model_key not in self.registry:
            print(f"[!] {model_key} not in env_map_conference.json. Skipping.")
            return False

        config = self.registry[model_key]
        
        in_flag = config.get("in_flag", "--input")
        out_flag = config.get("out_flag", "--output")
        weight_flag = config.get("weight_flag", "--weights")
        
        cmd = [
            config["python_bin"],
            config["script_path"],
            in_flag, input_path,
            out_flag, output_path,
        ]
        
        if config.get("weights"):
            cmd.extend([weight_flag, config["weights"]])

        if "extra_args" in config:
            cmd.extend(config["extra_args"])

        if config.get("routing") == "local" and mask_path:
            mask_flag = config.get("mask_flag", "--mask")
            cmd.extend([mask_flag, mask_path])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[*] SPAWNING SUBPROCESS: {model_key}")
        print(f"    -> CMD: {' '.join(cmd)}")

        # FIX: Use self.base_dir here
        working_dir = config.get("cwd", None)
        if working_dir:
            working_dir = str(self.base_dir / working_dir)
            
        result = subprocess.run(cmd, env=env, cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"[!] Subprocess {model_key} failed:\n{result.stderr}", file=sys.stderr)
            return False
            
        # STRICT I/O VERIFICATION
        if not os.path.exists(output_path):
            print(f"[!] FATAL: {model_key} reported success, but the output file was not found at {output_path}!", file=sys.stderr)
            print(f"    Check the {model_key} source code to see where it actually saves images.", file=sys.stderr)
            return False

        print(f"[+] Subprocess {model_key} succeeded and output verified. VRAM flushed.")
        return True
