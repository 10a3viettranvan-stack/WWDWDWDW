import json
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from utils import clean_text, extract_sections, logger

class MedicalDataset(Dataset):
    def __init__(self, df, tokenizer, max_input, max_output):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize Input (English)
        model_inputs = self.tokenizer(
            row["text"], 
            max_length=self.max_input, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Tokenize Target (Vietnamese)
        labels = self.tokenizer(
            row["target"], 
            max_length=self.max_output, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        labels_ids = labels["input_ids"].squeeze(0)
        
        # Thay thế pad token id trong labels bằng -100 để CrossEntropyLoss bỏ qua
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids
        }


class DataLoaderManager:
    """Quản lý việc tải và chia dữ liệu (Data Pipeline)."""
    def __init__(self, config):
        self.config = config

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Đọc dữ liệu tiếng Anh từ CSV và tiếng Việt từ TXT."""
        path = file_path or self.config["DATA"]["file_path"]
        
        # Thêm fallback tìm kiếm file
        if not Path(path).exists():
            for f in ["data/mtsamples_cleaned.csv", "data/mtsamples.csv", "mtsamples.csv"]:
                if Path(f).exists():
                    path = f
                    logger.info(f"Tìm thấy file thay thế: {path}")
                    break
            else:
                raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
                
        txt_path = self.config["DATA"].get("txt_file_path", "data/tom_tat_bang_tieng_viet.txt")
        logger.info(f"Tải Input (Tiếng Anh) từ: {path}")
        logger.info(f"Tải Target (Tiếng Việt) từ: {txt_path}")

        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1')
            
        logger.info(f"Số mẫu CSV ban đầu: {len(df)}")

        text_col = self.config["DATA"]["text_col"]
        if text_col not in df.columns:
            text_col = "transcription"
        
        df["text"] = df[text_col]
        df = df[["text"]].copy()
        
        # Đọc bản dịch tiếng Việt từ TXT
        vi_summaries = []
        try:
            if Path(txt_path).exists():
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt_content = f.read()
                records = re.split(r'BỆNH ÁN SỐ \d+', txt_content)[1:]
                for record in records:
                    tom_tat_vi = re.search(r'- Nội dung tóm tắt:\s*(.*)', record)
                    if tom_tat_vi:
                        vi_summaries.append(tom_tat_vi.group(1).strip())
                    else:
                        vi_summaries.append("không có tóm tắt")
            else:
                logger.warning(f"Không tìm thấy file {txt_path}. AI sẽ không học được tiếng Việt!")
        except Exception as e:
            logger.error(f"Lỗi đọc file TXT: {e}")
            
        # Ánh xạ Target vào DataFrame
        if len(vi_summaries) > 0:
            if len(vi_summaries) < len(df):
                vi_summaries.extend(["không có tóm tắt"] * (len(df) - len(vi_summaries)))
            df["target"] = vi_summaries[:len(df)]
        else:
            df["target"] = "không có tóm tắt"

        df = df.dropna()
        df["text"] = df["text"].apply(clean_text)
        df["target"] = df["target"].apply(clean_text)

        min_text_len = self.config["DATA"]["min_text_len"]
        min_sum_len = self.config["DATA"]["min_sum_len"]
        df = df[df["text"].str.len() >= min_text_len]
        df = df[df["target"].str.len() >= min_sum_len]

        max_samples = self.config["DATA"]["max_samples"]
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=self.config["DATA"]["random_state"])

        df = df.reset_index(drop=True)
        logger.info(f"Tổng số mẫu song ngữ (Anh -> Việt) sau lọc: {len(df)}")
        return df

    def split_dataset(self, df: pd.DataFrame) -> dict:
        """Chia dữ liệu thành train/val/test."""
        test_size = self.config["DATA"]["test_size"]
        val_size = self.config["DATA"]["val_size"]
        random_state = self.config["DATA"]["random_state"]

        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)

        logger.info(f"Đã chia: Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

        return {
            "train": train.reset_index(drop=True),
            "validation": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }
