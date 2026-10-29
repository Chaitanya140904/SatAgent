# core/profile/selector_conference.py

CONFERENCE_MODEL_TABLE = {
    "roi masking": {
        "Fidelity":   ["rs_sam", "cloud_net_plus", "yolov8_obb"],
        "Perception": ["rs_sam", "cloud_net_plus", "yolov8_obb"],
        "Hybrid":     ["rs_sam", "cloud_net_plus"]
    },
    "cloud removal": {
        "Fidelity":   ["dvpnet", "uncrtaints"],
        "Perception": ["emrdm", "cloud_mamba"],
        "Hybrid":     ["dvpnet", "emrdm"]
    },
    "dehazing": {
        "Fidelity":   ["rsdehamba", "rshazenet"],
        "Perception": ["dehazing_diffgan", "ridcp"],
        "Hybrid":     ["ridcp", "rsdehamba"]
    },
    "shadow correction": {
        "Fidelity":   ["shadowformer"],
        "Perception": ["enlightengan_rs"],
        "Hybrid":     ["shadowformer"]
    },
    "sensor denoising": {
        "Fidelity":   ["satmae", "nafnet_rs"],
        "Perception": ["diffusionsat"],
        "Hybrid":     ["nafnet_rs", "diffusionsat"]
    },
    "sensor deblurring": {
        "Fidelity":   ["pirn", "stripformer"],
        "Perception": ["blur_mamba", "deblurdiff"],
        "Hybrid":     ["stripformer", "blur_mamba"]
    },
    "destriping": {
        "Fidelity":   ["qgcn", "sst_net"],
        "Perception": ["repaint_rs"],
        "Hybrid":     ["qgcn"]
    },
    "compression artifact removal": {
        "Fidelity":   ["fbcnn", "nafnet_jpeg"],
        "Perception": ["swinir_car"],
        "Hybrid":     ["nafnet_jpeg", "swinir_car"]
    },
    "pan-sharpening": {
        "Fidelity":   ["pan_mamba", "adwm"],
        "Perception": ["mgdn", "premix"],
        "Hybrid":     ["pan_mamba", "mgdn"]
    },
    "multi-temporal fusion": {
        "Fidelity":   ["transfusion", "timesformer_rs"],
        "Perception": ["transfusion", "timesformer_rs"],
        "Hybrid":     ["transfusion"]
    },
    "super-resolution": {
        "Fidelity":   ["rep_mamba", "prithvi"],
        "Perception": ["satdiff", "transenet"],
        "Hybrid":     ["transenet", "swinir_gan"]  # <--- TransENet prioritized to bypass SwinIR
    }
}

class ConferenceSelector:
    @staticmethod
    def get_model(task: str, mode: str) -> str:
        return CONFERENCE_MODEL_TABLE.get(task, {}).get(mode, [""])[0]
