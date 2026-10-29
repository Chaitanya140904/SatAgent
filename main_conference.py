import argparse
import os
from core.profile.selector_conference import ConferenceSelector
from core.subprocess_router import ConferenceRouter

def run_conference_pipeline(image_path, profile_mode, output_dir, gpu_id):
    os.makedirs(output_dir, exist_ok=True)
    router = ConferenceRouter()
    
    # Hardware/Physics logic cascade (Hardcoded for testing parallel track)
    # In production, replace this with the parsed JSON from Agent 1
    plan = [
        {"task": "cloud removal", "routing": "local"},
        {"task": "dehazing", "routing": "global"},
        {"task": "super-resolution", "routing": "global"}
    ]
    
    current_img = image_path
    
    for step_idx, step in enumerate(plan):
        model_name = ConferenceSelector.get_model(step["task"], profile_mode)
        
        if not model_name:
            print(f"[-] No model mapped for {step['task']} in {profile_mode} mode. Skipping.")
            continue
            
        out_img = os.path.join(output_dir, f"step_{step_idx+1}_{model_name}.png")
        
        success = router.execute_expert(
            model_key=model_name,
            input_path=current_img,
            output_path=out_img,
            gpu_id=gpu_id
        )
        
        if success:
            current_img = out_img
        else:
            print(f"[!] Pipeline halted at phase: {step['task']}")
            break

    print(f"\n==================================================")
    print(f"🏁 CONFERENCE PIPELINE COMPLETE")
    print(f"Final output: {current_img}")
    print(f"==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", default="Fidelity", choices=["Fidelity", "Perception", "Hybrid"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=1)
    args = parser.parse_args()
    
    run_conference_pipeline(args.image, args.mode, args.output_dir, args.gpu_id)
