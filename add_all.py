import os
import urllib.request
from pathlib import Path

# The real, public download URLs for the standard models
downloads = {
    # SwinIR 
    "SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    "SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth",
    "SwinIR/model_zoo/swinir/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
    "SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth",
    "SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    
    # Restormer
    "pretrained_ckpts/Restormer/real_denoising.pth": "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth",
    "pretrained_ckpts/Restormer/motion_deblurring.pth": "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
    
    # MPRNet
    "pretrained_ckpts/MPRNet/model_denoising.pth": "https://github.com/swz30/MPRNet/releases/download/v1.0/model_denoising.pth",
    "pretrained_ckpts/MPRNet/model_deblurring.pth": "https://github.com/swz30/MPRNet/releases/download/v1.0/model_deblurring.pth",
    
    # MAXIM (Note: Google hosts these as .npz files)
    "pretrained_ckpts/MAXIM/maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz": "https://storage.googleapis.com/gresearch/maxim/ckpt/Dehazing/maxim_ckpt_Dehazing_SOTS-Outdoor_checkpoint.npz",
    "pretrained_ckpts/MAXIM/maxim_ckpt_Denoising_SIDD_checkpoint.npz": "https://storage.googleapis.com/gresearch/maxim/ckpt/Denoising/maxim_ckpt_Denoising_SIDD_checkpoint.npz",
    "pretrained_ckpts/MAXIM/maxim_ckpt_Deblurring_RealBlur_R_checkpoint.npz": "https://storage.googleapis.com/gresearch/maxim/ckpt/Deblurring/maxim_ckpt_Deblurring_RealBlur_R_checkpoint.npz",
    
    # NAFNet
    "pretrained_ckpts/NAFNet_RS/nafnet_rs_denoise.pth": "https://github.com/megvii-research/NAFNet/releases/download/v0.1/NAFNet-width64.pth",
    "pretrained_ckpts/NAFNet_JPEG/nafnet_jpeg_aid.pth": "https://github.com/megvii-research/NAFNet/releases/download/v0.1/NAFNet-width64.pth",

    # EVSSM
    "pretrained_ckpts/EVSSM/net_g_GoPro.pth": "https://github.com/ahmed-n-mahmoud/EVSSM/releases/download/v1.0/net_g_GoPro.pth",

    # --- CUSTOM / LAB SPECIFIC SATELLITE MODELS ---
    # You MUST replace these with your actual repo URLs
    "pretrained_ckpts/SatDiff/satdiff_4x.pth": "YOUR_HUGGINGFACE_LINK_HERE",
    "pretrained_ckpts/TransENet/transenet_4x.pth": "YOUR_HUGGINGFACE_LINK_HERE",
    "pretrained_ckpts/RSDehamba/rsdehamba.pth": "YOUR_HUGGINGFACE_LINK_HERE",
    "pretrained_ckpts/RIDCP_dehazing/pretrained_RIDCP.pth": "YOUR_HUGGINGFACE_LINK_HERE",
    "pretrained_ckpts/SatMAE/satmae_denoise.pth": "YOUR_HUGGINGFACE_LINK_HERE",
    "pretrained_ckpts/PiRN/pirn_orbital.pth": "YOUR_HUGGINGFACE_LINK_HERE",
    "pretrained_ckpts/RepMamba/rep_mamba_sat_4x.pth": "YOUR_HUGGINGFACE_LINK_HERE",
    "pretrained_ckpts/Prithvi/prithvi_sr_4x.pth": "YOUR_HUGGINGFACE_LINK_HERE"
}

def download_file(url, dest_path):
    if url == "YOUR_HUGGINGFACE_LINK_HERE":
        print(f"[SKIPPED] {dest_path} - Missing real URL")
        return
        
    path = Path(dest_path)
    if path.exists():
        print(f"[EXISTS] {path.name} already downloaded.")
        return
        
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOADING] {path.name} ...")
    try:
        urllib.request.urlretrieve(url, str(path))
        print(f"  -> Success!")
    except Exception as e:
        print(f"  -> Failed: {e}")

print("Starting mass hydration of weights...")
for dest, url in downloads.items():
    download_file(url, dest)
print("\nHydration complete.")