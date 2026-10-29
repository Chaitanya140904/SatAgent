import json
from pathlib import Path

manifest_path = Path("weights_manifest.json")

# Dictionary mapping the required checkpoint paths to their download URLs
# (Replace these dummy URLs with your actual HuggingFace/Release links)
url_mapping = {
    "pretrained_ckpts/TransENet/transenet_4x.pth": "https://huggingface.co/your-repo/TransENet/resolve/main/transenet_4x.pth",
    "pretrained_ckpts/RIDCP_dehazing/pretrained_RIDCP.pth": "https://huggingface.co/your-repo/RIDCP/resolve/main/pretrained_RIDCP.pth",
    "pretrained_ckpts/ShadowFormer/shadowformer.pth": "https://huggingface.co/your-repo/ShadowFormer/resolve/main/shadowformer.pth",
    "pretrained_ckpts/PanMamba/panmamba.pth": "https://huggingface.co/your-repo/PanMamba/resolve/main/panmamba.pth",
    "pretrained_ckpts/Rep-Mamba/repmamba.pth": "https://huggingface.co/your-repo/RepMamba/resolve/main/repmamba.pth"
}

if manifest_path.exists():
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    for ckpt, data in manifest.items():
        # Match the end of the path to our mapping
        for mapped_path, url in url_mapping.items():
            if ckpt.endswith(mapped_path.split('/')[-1]):
                manifest[ckpt]["url"] = url
                print(f"Patched URL for: {ckpt}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print("Manifest patched successfully!")
else:
    print("weights_manifest.json not found. Run the scan command first.")