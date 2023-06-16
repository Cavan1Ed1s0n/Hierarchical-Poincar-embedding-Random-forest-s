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


### References
[What is Poincaré Embeddings](https://medium.com/@sri33/explaining-poincar%C3%A9-embeddings-d7cb9e4a2bbf)

[Hyperbolic-Learning](https://github.com/drewwilimitis/hyperbolic-learning)

[Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/pdf/1705.08039.pdf)

[Explaination for Poincaré Embeddings for Learning Hierarchical Representations](https://www.slideshare.net/daynap1204/poincare-embeddings-for-learning-hierarchical representations)

[KGE-CL: Contrastive Learning of Tensor Decomposition Based Knowledge Graph Embeddings](https://arxiv.org/pdf/2112.04871.pdf)

[Implement Poincare Embeddings](https://rare-technologies.com/implementing-poincare-embeddings/)

[Code run on colab](https://colab.research.google.com/drive/13-nOoOzFjiDlFOQEepc91Nj3t45PIjQw#scrollTo=N7kwZzX89gzh)

[Resource](https://drive.google.com/drive/u/0/folders/1RfYoT_yNK9hMNrWsRaYEDqhpWTKfOQ5Y)




