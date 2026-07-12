import json
import os
from pathlib import Path
import torch

# Mặc định sử dụng 5 luồng theo yêu cầu
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
torch.set_num_threads(5)

DEFAULT_CONFIG = {
    "DATA": {
        "file_path": "data/mtsamples.csv",
        "txt_file_path": "data/tom_tat_bang_tieng_viet.txt",
        "text_col": "transcription",
        "summary_col": "description",
        "max_samples": 5000,
        "min_text_len": 50,
        "min_sum_len": 5,
        "test_size": 0.15,
        "val_size": 0.10,
        "random_state": 42,
        "fallback_summary_cols": ["keywords", "medical_specialty"]
    },
    "MODEL": {
        "model_name": "facebook/bart-base",  # Có thể đổi thành distilbart nếu muốn nhẹ hơn
        "max_input": 256,
        "max_output": 128
    },
    "TRAIN": {
        "output_dir": "./models/marian_translator",
        "epochs": 5,
        "lr": 5e-5,
        "batch_size": 8,                # Tăng batch size
        "accumulation_steps": 2         # Thêm accumulation để mô phỏng batch lớn hơn
    },
    "GENERATE": {
        "max_len": 128,
        "num_beams": 4
    }
}

CONFIG = DEFAULT_CONFIG.copy()

def load_config(config_path: str = "config.json"):
    """Tải cấu hình từ file JSON."""
    global CONFIG
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for key, val in DEFAULT_CONFIG.items():
                if key not in loaded:
                    loaded[key] = val
                else:
                    for k, v in val.items():
                        if k not in loaded[key]:
                            loaded[key][k] = v
            CONFIG = loaded
        except Exception:
            CONFIG = DEFAULT_CONFIG
    else:
        CONFIG = DEFAULT_CONFIG
    return CONFIG