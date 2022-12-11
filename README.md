# Hierarchical-Poincar-embedding-Random-forest-s
### Reference
Contrastive Learning: https://arxiv.org/abs/2112.04871
Lars' Blog - Implementing Poincaré Embeddings in PyTorch (lars76.github.io): https://lars76.github.io/2020/07/24/implementing-poincare-embedding.html

Sử dụng hierarchical/Poincaré embeddings để biểu diễn mô hình **Random forest (các tree/graph).**

Dùng **[Contrastive Learning](https://arxiv.org/abs/2112.04871) để embedding tree: các node cùng cây link trực tiếp nhau thì gần nhau, các node dùng cùng feature thì gần nhau.**

Mỗi điểm trong không gian mới kết quả của phép nhúng 1 node trên cây (1 phần data ban đầu).

Mỗi điểm trong data ban đầu được map với nhiều điểm trong không gian mới (nodes mà điểm data đó đi qua trên cây).

Ngữ cảnh áp dụng:

- data ban đầu là dữ liệu dạng bảng, có missing
- Tìm cách đưa dữ liệu này thành dạng liên tục làm input cho ANN.
