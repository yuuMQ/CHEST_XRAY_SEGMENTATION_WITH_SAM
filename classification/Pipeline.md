# Vision Transformer (ViT)

![image](ViT_pipeline.png)

**Input**: Ảnh RGB kích thước cố định (ví dụ: `224x224x3`)

**Các bước xử lý**:

1. **Chia ảnh thành patch**: ví dụ `16x16` → Tổng cộng `N = (224/16)^2 = 196` patch
2. **Flatten mỗi patch** → vector kích thước `P`
3. **Linear Projection** → mỗi patch thành vector nhúng (embedding) kích thước `D` (ví dụ: `768`)
4. **Thêm token [CLS]** đặc biệt vào đầu chuỗi patch
5. **Cộng positional embedding** (learnable)
6. Đưa vào `L` tầng Transformer Encoder, mỗi tầng gồm:
    - Multi-Head Self Attention
    - Layer Normalization + Skip connection
    - MLP (Feedforward Network)
7. **Lấy token [CLS]** sau Transformer → biểu diễn toàn ảnh
8. Qua MLP head → Softmax → Phân loại

---
