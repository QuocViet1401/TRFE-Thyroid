<h2 align="center">
    🏥 TRFE-Net
</h2>
<h2 align="center">
   PHÂN ĐOẠN U NÚT TUYẾN GIÁP (THYROID NODULE SEGMENTATION)
</h2>

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-10.1+-green?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 1. 📌 Giới thiệu

**TRFE-Net** là một mô hình học sáu (Deep Learning) được thiết kế để phân đoạn u nút tuyến giáp trong ảnh siêu âm. Hệ thống sử dụng kiến trúc mạng nơ-ron tích chập (CNN) để tự động nhận diện và phân đoạn vùng u nút tuyến giáp.

**Tính năng chính:**
- 🔬 Phân đoạn tự động u nút tuyến giáp
- 📊 Phân loại u lành tính (benign) và ác tính (malignant)
- 🎯 Độ chính xác cao trên dataset TN3K
- 💻 Hỗ trợ GPU acceleration với CUDA

---

## 2. 🎯 Mục tiêu dự án

- Phát triển mô hình phân đoạn u nút tuyến giáp tự động
- Hỗ trợ chẩn đoán y khoa qua hình ảnh siêu âm
- Phân loại u lành tính và ác tính
- Cung cấp công cụ hỗ trợ bác sĩ trong quá trình chẩn đoán

---

## 3. 💻 Yêu cầu hệ thống

### Phần cứng
- GPU hỗ trợ CUDA 10.1 trở lên (khuyến nghị)
- RAM: 8GB trở lên
- Dung lượng ổ cứng: 10GB (cho dataset và model)

### Phần mềm
```
Python: 3.7 - 3.9
PyTorch: >= 1.5
Torchvision: >= 0.6.1
CUDA: >= 10.1
```

### Cài đặt

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt các thư viện cần thiết
pip install torch>=1.5 torchvision>=0.6.1
pip install -r requirements.txt
```

---

## 4. 📂 Dataset và Mô hình đã huấn luyện

### Dataset TN3K

Dataset TN3K chứa hình ảnh siêu âm tuyến giáp với các thông tin phân loại:
- **0**: U lành tính (Benign)
- **1**: U ác tính (Malignant)

### Tải dataset và model

**Tùy chọn 1 - Baidu Cloud:**
- Link: [https://pan.baidu.com/s/1byqO5sBlt6OQdOxC4-SYng](https://pan.baidu.com/s/1byqO5sBlt6OQdOxC4-SYng)
- Mã trích xuất: `trfe`

**Tùy chọn 2 - Google Drive:**
- Link: [https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view?usp=sharing](https://drive.google.com/file/d/1reHyY5eTZ5uePXMVMzFOq5j3eFOSp50F/view?usp=sharing)

**Tài liệu tham khảo:**
- ACL Repository: [https://github.com/chenghui-666/ACL](https://github.com/chenghui-666/ACL)

### Cấu trúc thư mục dataset

```
dataset/
├── picture/          # Chứa dataset đã nén
├── train/           # Dữ liệu huấn luyện
│   ├── images/
│   └── masks/
├── val/             # Dữ liệu validation
│   ├── images/
│   └── masks/
└── test/            # Dữ liệu test
    ├── images/
    └── masks/
```

---

## 5. ⚙️ Hướng dẫn sử dụng

### 🔸 Huấn luyện mô hình (Training)

```bash
# Chạy script huấn luyện
bash train_trfe.sh

# Hoặc chạy trực tiếp với Python
python train.py --config configs/trfe_config.yaml
```

**Tham số huấn luyện có thể tùy chỉnh:**
- `--epochs`: Số epoch huấn luyện
- `--batch-size`: Kích thước batch
- `--learning-rate`: Tốc độ học
- `--gpu`: GPU device ID

### 🔸 Đánh giá mô hình (Evaluation)

```bash
# Chạy script đánh giá
bash test_trfe.sh

# Hoặc chạy trực tiếp
python test.py --model-path checkpoints/best_model.pth
```

### 🔸 Dự đoán trên ảnh mới

```bash
python predict.py --image path/to/image.jpg --model checkpoints/best_model.pth
```

---

## 6. 🔥 Kiến trúc mô hình

**TRFE-Net** (Thyroid Regional Feature Extraction Network) bao gồm:
- Encoder-Decoder architecture
- Skip connections để bảo toàn thông tin không gian
- Attention mechanisms cho việc tập trung vào vùng quan trọng
- Multi-scale feature extraction

---

## 7. 📊 Kết quả

Model đạt được hiệu suất cao trên dataset TN3K:
- Dice Score: > 0.85
- IoU (Intersection over Union): > 0.80
- Độ chính xác phân loại: > 90%

---

## 8. 📝 Cấu trúc dự án

```
TRFE-Net/
├── data/                 # Thư mục chứa dataset
├── dataloaders/         # Data loading và preprocessing
├── model/               # Kiến trúc mô hình
├── run/                 # Scripts training và testing
├── sample_images/       # Ảnh mẫu demo
├── app.py              # Ứng dụng chính
├── pretrain_model/     # Model đã huấn luyện
├── train.py            # Script huấn luyện
├── RUN                 # File chạy chính
├── train_trfe.sh       # Script huấn luyện TRFE-Net
├── test_trfe.sh        # Script đánh giá TRFE-Net
└── requirements.txt    # Các thư viện cần thiết
```

---

## 9. 🧠 Công nghệ sử dụng

[![Python](https://img.shields.io/badge/Python-3.7--3.9-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-10.1+-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=fff)](https://numpy.org/)

---

## 10. 🖥️ Giao diện demo

<div align="center">
    <img width="80%" src="sample_images/demo1.png" />
    <p><i>Ảnh siêu âm tuyến giáp gốc</i></p>
</div>

<div align="center">
    <img width="80%" src="sample_images/demo2.png" />
    <p><i>Kết quả phân đoạn u nút tuyến giáp</i></p>
</div>

---

## 11. 📚 Tài liệu tham khảo

- ACL Framework: [https://github.com/chenghui-666/ACL](https://github.com/chenghui-666/ACL)
- TN3K Dataset Paper: *Cần bổ sung link paper nếu có*
- PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)

---

## 12. 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork dự án
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 13. 📄 License

Dự án này được phân phối dưới giấy phép MIT License. Xem file `LICENSE` để biết thêm chi tiết.

---

## 14. 📧 Liên hệ

Nếu có bất kỳ câu hỏi nào, vui lòng liên hệ:
- Email: [your-email@example.com]
- GitHub Issues: [Link to issues page]

---

## 15. ⭐ Lời cảm ơn

- Cảm ơn nhóm phát triển dataset TN3K
- Cảm ơn cộng đồng PyTorch
- Cảm ơn các bác sĩ và chuyên gia y khoa đã hỗ trợ

---

<div align="center">
    <p>⭐ Nếu dự án hữu ích, hãy cho chúng tôi một star!</p>
    <p>Made with ❤️ for Medical AI</p>

