# Hierarchical-Poincare-embedding-Random-forest
### 
Sử dụng hierarchical/Poincaré embeddings để biểu diễn mô hình **Random forest (các tree/graph).**

Dùng **[Contrastive Learning](https://arxiv.org/abs/2112.04871) để embedding tree: các node cùng cây link trực tiếp nhau thì gần nhau, các node dùng cùng feature thì gần nhau.**

Mỗi điểm trong không gian mới kết quả của phép nhúng 1 node trên cây (1 phần data ban đầu).

Mỗi điểm trong data ban đầu được map với nhiều điểm trong không gian mới (nodes mà điểm data đó đi qua trên cây).

Ngữ cảnh áp dụng:

- data ban đầu là dữ liệu dạng bảng, có missing
- Tìm cách đưa dữ liệu này thành dạng liên tục làm input cho ANN.

[Lars' Blog - Implementing Poincaré Embeddings in PyTorch (lars76.github.io)](https://lars76.github.io/2020/07/24/implementing-poincare-embedding.html) (nhớ chặn chuẩn các vector <1)



https://www.slideshare.net/daynap1204/poincare-embeddings-for-learning-hierarchical-representations

https://arxiv.org/pdf/1705.08039.pdf

https://arxiv.org/pdf/2112.04871.pdf

https://drive.google.com/drive/u/0/folders/1t-QNbY7NS8JdcjEcjM6dVHptiPd5yAvF

https://colab.research.google.com/drive/13-nOoOzFjiDlFOQEepc91Nj3t45PIjQw#scrollTo=N7kwZzX89gzh

https://github.com/drewwilimitis/hyperbolic-learning
