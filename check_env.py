import os

import psutil
import torch


def main() -> None:
    """يطبع أرقام فحص البيئة المطلوبة في المرحلة 0."""
    available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)

    print("البيئة جاهزة")
    print(f"إصدار PyTorch: {torch.__version__}")
    print(f"عدد أنوية/خيوط المعالج: {os.cpu_count()}")
    print(f"الرامة المتاحة: {available_ram_mb:.2f} ميجابايت")
    print(f"CUDA متاح؟: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
