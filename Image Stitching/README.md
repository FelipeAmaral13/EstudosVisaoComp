# Image Stitching

Imagens Originais:

![foto1](https://user-images.githubusercontent.com/5797933/85855928-69b59100-b78d-11ea-94e1-2ead282a3d0c.jpeg)
![foto2](https://user-images.githubusercontent.com/5797933/85855923-68846400-b78d-11ea-92c0-fa47ccc2bbcb.jpeg)

Seguintes passos foram desenvolvidos para esse projeto:

* Calculcar os orb-keypoints e seus descritores para as imagens originais.

![1](https://user-images.githubusercontent.com/5797933/85856524-8c947500-b78e-11ea-9cbf-02ece945a5b7.png)

* Calcular a distância entre cada desdritores da primeira imagem em relação a segunda.

![2](https://user-images.githubusercontent.com/5797933/85856754-ee54df00-b78e-11ea-8bb5-8982de18f4fe.png)

* Selecionar os matchs. 
* Rodar oRANSAC para estimar a homografia.
* Fazer o stitch das imagens.

![3](https://user-images.githubusercontent.com/5797933/85856623-b8175f80-b78e-11ea-9e2a-18db53f40110.png)
