from pathlib import Path
import warnings
import torch
import pytesseract

warnings.filterwarnings("ignore", category=FutureWarning)

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Tự động tìm modellm3 ở thư mục cha hoặc ngay trong demo2
MODEL_DIR = (
    PROJECT_DIR / "modellm3"
    if (PROJECT_DIR / "modellm3").exists()
    else BASE_DIR / "modellm3"
)
INPUT_DIR = BASE_DIR / "demo_input"
OUTPUT_DIR = BASE_DIR / "layoutlm3_output"
ORIENTED_DIR = OUTPUT_DIR / "oriented_images"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ORIENTED_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình Tesseract & Thiết bị
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OCR_LANG = "vie+eng"

# Từ khóa hậu xử lý STORE
STORE_STOP_PHRASES = [
    "hoa don ban hang",
    "hoa don thanh toan",
    "hoa don",
    "ban hang",
    "ma so thue",
    "dien thoai",
    "ngay mua",
    "ngay ban",
    "ten hang",
    "mat hang",
    "thanh tien",
    "quay"
]

# Từ khóa hậu xử lý TOTAL
TOTAL_KEYWORDS = [
    "tong tien phai t.toan",
    "tong tien phai",
    "tong cong",
    "teng ceng",
    "tong thanh toan",
    "tong tien thanh toan",
    "tong tien",
    "thanh tien",
    "total",
    "tong so thanh toan",
    "grand total"
]

EXCLUDE_MONEY_KEYWORDS = [
    "tien khach tra",
    "khach tra",
    "tien mat",
    "tien tra lai",
    "tra lai",
    "tien thua",
    "da giam",
    "a giam",
    "change",
    "cash tendered"
]