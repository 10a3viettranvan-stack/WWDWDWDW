import sys
import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from transformers import BartTokenizer, BartForConditionalGeneration
from config import load_config
from utils import ensure_dirs, logger
from data_loader import DataLoaderManager, MedicalDataset
from trainer import ModelTrainer
from predictor import Predictor
from translator import TranslatorMemory

def setup_environment():
    """Khởi tạo thư mục và cấu hình."""
    ensure_dirs("data", "models", "output", "logs")
    return load_config("config.json")

def init_model_and_tokenizer(config, device, load_pretrained=False):
    """Khởi tạo hoặc tải tokenizer và model BART."""
    output_dir = Path(config["TRAIN"]["output_dir"])
    model_name = config["MODEL"]["model_name"]
    
    if load_pretrained:
        if not (output_dir / "model_best").exists():
            logger.error("❌ Chưa có mô hình. Hãy huấn luyện trước.")
            return None, None
        logger.info(f"Đang tải model và tokenizer từ: {output_dir / 'model_best'}")
        tokenizer = BartTokenizer.from_pretrained(str(output_dir / "model_best"))
        model = BartForConditionalGeneration.from_pretrained(str(output_dir / "model_best")).to(device)
    else:
        logger.info(f"Đang tải pretrained model: {model_name}")
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    return model, tokenizer

def run_translate(config):
    """Chức năng 1: Dịch dữ liệu Train (sử dụng Translation Memory)."""
    logger.info("=" * 60)
    logger.info("CHỨC NĂNG: DỊCH VÀ TÓM TẮT BỆNH ÁN")
    logger.info("=" * 60)
    
    file_path = config["DATA"]["file_path"]
    translator = TranslatorMemory()  # Tự động load cache từ txt nếu chưa có
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception:
        df = pd.read_csv(file_path, encoding='latin-1')

    output_file = 'tom_tat_benh_an_tieng_viet.txt'
    logger.info(f"Đang dịch và lưu tại: {output_file}")
    
    # Dịch batch để tăng tốc
    # Chúng ta sẽ collect các cột cần dịch
    col_specialty = df['medical_specialty'].astype(str).tolist()
    col_sample = df['sample_name'].astype(str).tolist()
    col_desc = df['description'].astype(str).tolist()
    
    # Dịch batch từng cột (nếu translator hỗ trợ)
    translated_specialty = translator.translate_batch(col_specialty)
    translated_sample = translator.translate_batch(col_sample)
    translated_desc = translator.translate_batch(col_desc)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== DANH SÁCH TÓM TẮT {len(df)} BỆNH ÁN ===\n\n")
        for index in range(len(df)):
            chuyen_khoa = translated_specialty[index]
            ten_mau = translated_sample[index]
            tom_tat = translated_desc[index]
            
            f.write(f"BỆNH ÁN SỐ {index + 1}\n")
            f.write(f"- Chuyên khoa: {chuyen_khoa}\n")
            f.write(f"- Tên mẫu: {ten_mau}\n")
            f.write(f"- Nội dung tóm tắt: {tom_tat}\n")
            f.write("-" * 60 + "\n\n")
            
            if (index + 1) % 50 == 0:
                logger.info(f"Đã xử lý {index + 1}/{len(df)} bệnh án.")
    
    logger.info("🎉 Dịch hoàn tất!")

def run_train(config, device, resume=False):
    """Chức năng 2 & 3: Huấn luyện / Resume với DataLoader tối ưu."""
    logger.info("=" * 60)
    logger.info("CHỨC NĂNG: HUẤN LUYỆN MÔ HÌNH" if not resume else "CHỨC NĂNG: RESUME MÔ HÌNH")
    logger.info("=" * 60)

    data_mgr = DataLoaderManager(config)
    df = data_mgr.load_data()
    dataset = data_mgr.split_dataset(df)

    output_dir = Path(config["TRAIN"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        model, tokenizer = init_model_and_tokenizer(config, device, load_pretrained=False)
    else:
        model, tokenizer = init_model_and_tokenizer(config, device, load_pretrained=True)
        if not model: return

    train_dataset = MedicalDataset(dataset["train"], tokenizer, config["MODEL"]["max_input"], config["MODEL"]["max_output"])
    val_dataset = MedicalDataset(dataset["validation"], tokenizer, config["MODEL"]["max_input"], config["MODEL"]["max_output"])

    # DataLoader với num_workers và pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["TRAIN"]["batch_size"],
        shuffle=True,
        num_workers=4,          # Tăng số worker
        pin_memory=True,        # Tăng tốc truyền lên GPU
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["TRAIN"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    trainer = ModelTrainer(model, config, tokenizer, device)
    
    if resume:
        resume_path = output_dir / "checkpoint_best.pt"
        trainer.train(train_loader, val_loader, resume_from=resume_path, extra_epochs=10)
    else:
        trainer.train(train_loader, val_loader)

def run_demo(config, device):
    """Chức năng 4: Demo tương tác (sử dụng predictor)."""
    logger.info("=" * 60)
    logger.info("CHỨC NĂNG: DEMO TÓM TẮT TƯƠNG TÁC")
    logger.info("=" * 60)

    model, tokenizer = init_model_and_tokenizer(config, device, load_pretrained=True)
    if not model: return

    predictor = Predictor(model, config, tokenizer)
    translator = TranslatorMemory()
    
    data_mgr = DataLoaderManager(config)
    df = data_mgr.load_data()

    print("\n" + "=" * 70)
    print("📋 DANH SÁCH BỆNH ÁN")
    for idx, row in df.head(20).iterrows():
        desc = str(row.get('summary', ''))[:80] + "..."
        print(f"[{idx:3d}] | {desc}")
    print("... (hiển thị 20 mẫu đầu)")
    
    while True:
        user_input = input("\n🔢 Nhập số thứ tự bệnh án (hoặc 'exit' để thoát): ").strip()
        if user_input.lower() == 'exit':
            break
        
        try:
            idx = int(user_input)
            if idx < 0 or idx >= len(df):
                print(f"❌ Vui lòng nhập từ 0 đến {len(df)-1}")
                continue
            
            text = df.iloc[idx]["text"]
            print("\n⏳ Đang tóm tắt...")
            result = predictor.predict(text)
            
            print("\n📋 KẾT QUẢ GỐC (TIẾNG ANH):")
            print(f"  📝 Tóm tắt chung: {result['summary']}")
            print(f"  🩺 Triệu chứng: {result['symptoms']}")
            print(f"  🔬 Chẩn đoán: {result['diagnosis']}")
            print(f"  💊 Điều trị: {result['prescription']}")
            
            print("\n🇻🇳 KẾT QUẢ DỊCH (TIẾNG VIỆT):")
            # Dịch từng phần, sử dụng cache
            print(f"  📝 Tóm tắt chung: {translator.translate(result['summary'])}")
            print(f"  🩺 Triệu chứng: {translator.translate(result['symptoms'])}")
            print(f"  🔬 Chẩn đoán: {translator.translate(result['diagnosis'])}")
            print(f"  💊 Điều trị: {translator.translate(result['prescription'])}")
            print("-" * 70)
            
        except ValueError:
            print("❌ Lỗi: Cần nhập một số hợp lệ!")

def run_evaluate(config, device):
    """Chức năng 5: Đánh giá ROUGE (dùng batch predict)."""
    model, tokenizer = init_model_and_tokenizer(config, device, load_pretrained=True)
    if not model: return
    
    data_mgr = DataLoaderManager(config)
    df = data_mgr.load_data()
    dataset = data_mgr.split_dataset(df)
    
    predictor = Predictor(model, config, tokenizer)
    scores = predictor.evaluate(dataset["test"], batch_size=config["TRAIN"]["batch_size"])
    
    out_path = Path("output/evaluation.json")
    out_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    logger.info(f"Đã lưu kết quả tại: {out_path}")

def interactive_menu():
    """Giao diện Menu tương tác trên Terminal."""
    config = setup_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Sử dụng thiết bị: {device}")
    
    while True:
        print("\n" + "=" * 60)
        print(" CHƯƠNG TRÌNH TÓM TẮT VÀ DỊCH BỆNH ÁN (OOP VER)".center(60))
        print("=" * 60)
        print(" 1. Dịch dữ liệu sang Tiếng Việt (Sử dụng Cache Memory)")
        print(" 2. Huấn luyện mô hình từ đầu")
        print(" 3. Tiếp tục huấn luyện (Resume) từ checkpoint")
        print(" 4. Chạy Demo tương tác (Dự đoán & Dịch kết quả)")
        print(" 5. Đánh giá mô hình (ROUGE Score)")
        print(" 0. Thoát")
        print("=" * 60)
        
        choice = input("👉 Vui lòng chọn chức năng (0-5): ").strip()
        
        if choice == "1":
            run_translate(config)
        elif choice == "2":
            run_train(config, device, resume=False)
        elif choice == "3":
            run_train(config, device, resume=True)
        elif choice == "4":
            run_demo(config, device)
        elif choice == "5":
            run_evaluate(config, device)
        elif choice == "0" or choice.lower() == "exit":
            print("👋 Tạm biệt!")
            sys.exit(0)
        else:
            print("❌ Lựa chọn không hợp lệ, vui lòng thử lại!")

if __name__ == "__main__":
    interactive_menu()