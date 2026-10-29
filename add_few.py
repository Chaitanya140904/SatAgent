import os
import urllib.request
from pathlib import Path
try:
    import gdown
except ImportError:
    print("[!] Error: 'gdown' is not installed. Run: pip install gdown")
    exit(1)

# ---------------------------------------------------------
# THE GOOGLE DRIVE TARGETS
# Replace the 'YOUR_FILE_ID_HERE' placeholders with the actual Google Drive IDs
# ---------------------------------------------------------
downloads = {
    # MPRNet
    "pretrained_ckpts/MPRNet/model_denoising.pth": "https://drive.google.com/file/d/1LODPt9kYmxwU98g96UrRA0_Eh5HYcsRw",
    "pretrained_ckpts/MPRNet/model_deblurring.pth": "https://drive.google.com/file/d/1QwQUVbk6YVOJViCsOKYNykCsdJSVGRtb",
    
    # NAFNet
    "pretrained_ckpts/NAFNet_RS/nafnet_rs_denoise.pth": "https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR",
    "pretrained_ckpts/NAFNet_JPEG/nafnet_jpeg_aid.pth": "https://drive.google.com/file/d/1yy0oPNJjJxfaEmO0pfPW_TpeoCotYkuO",
    
    # EVSSM
    "pretrained_ckpts/EVSSM/net_g_GoPro.pth": "https://drive.google.com/file/d/1SN5CoNEiBa_pXojD4K661FbYkYFSyLFs",

    # --- CUSTOM / LAB SPECIFIC SATELLITE MODELS ---
    "pretrained_ckpts/SatDiff/satdiff_4x.pth": "https://drive.google.com/uc?id=YOUR_SATDIFF_ID",
    "pretrained_ckpts/TransENet/transenet_4x.pth": "https://drive.google.com/file/d/19nH1Plh2M-Z47iXG0-Ghq-Orh33n787w",
    "pretrained_ckpts/RSDehamba/rsdehamba.pth": "https://drive.google.com/uc?id=YOUR_RSDEHAMBA_ID",
    "pretrained_ckpts/RIDCP_dehazing/pretrained_RIDCP.pth": "https://drive.google.com/uc?id=YOUR_RIDCP_ID",
    "pretrained_ckpts/SatMAE/satmae_denoise.pth": "https://zenodo.org/records/7369797/files/fmow_pretrain.pth?download=1",
    "pretrained_ckpts/PiRN/pirn_orbital.pth": "https://drive.google.com/uc?id=YOUR_PIRN_ID",
    "pretrained_ckpts/RepMamba/rep_mamba_sat_4x.pth": "https://drive.google.com/uc?id=YOUR_REPMAMBA_ID",
    "pretrained_ckpts/Prithvi/prithvi_sr_4x.pth": "https://drive.google.com/uc?id=YOUR_PRITHVI_ID",

    # --- MAXIM (Google Cloud Storage) ---
    # GCS URLs can be finicky. If these 404 again, you may need to use gsutil or check if the bucket was locked.
    "pretrained_ckpts/MAXIM/maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz": "https://storage.googleapis.com/gresearch/maxim/ckpt/Dehazing/maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz",
    "pretrained_ckpts/MAXIM/maxim_ckpt_Denoising_SIDD_checkpoint.npz": "https://storage.googleapis.com/gresearch/maxim/ckpt/Denoising/maxim_ckpt_Denoising_SIDD_checkpoint.npz",
    "pretrained_ckpts/MAXIM/maxim_ckpt_Deblurring_RealBlur_R_checkpoint.npz": "https://storage.googleapis.com/gresearch/maxim/ckpt/Deblurring/maxim_ckpt_Deblurring_RealBlur_R_checkpoint.npz"
}

def download_file(url, dest_path):
    if "YOUR_" in url:
        print(f"[SKIPPED] {dest_path} - Needs real Google Drive File ID")
        return
        
    path = Path(dest_path)
    if path.exists():
        print(f"[EXISTS] {path.name} already downloaded.")
        return
        
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOADING] {path.name} ...")
    
    try:
        # Use gdown for Google Drive links, fallback to urllib for direct GCS links
        if "drive.google.com" in url:
            # fuzzy=True forces gdown to parse through the HTML warning pages
            gdown.download(url, str(path), quiet=False, fuzzy=True)
        else:
            urllib.request.urlretrieve(url, str(path))
            
        if path.exists():
            print(f"  -> Success!")
        else:
            print(f"  -> Failed: File was not created.")
    except Exception as e:
        print(f"  -> Failed: {e}")

print("Starting GDown hydration of missing weights...")
for dest, url in downloads.items():
    download_file(url, dest)
print("\nHydration complete.")