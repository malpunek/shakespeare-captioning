# Unsupervised Shakespeare Style Captioning

Following a rapid development in state-of-the-art image capitoning models the goal of this master thesis is to train a neural network inspired by [SemStyle paper](https://arxiv.org/abs/1805.07030) to achieve image captioning in Shakespearian style. For instance given an image:

[![Tommy Wiseau](https://bi.im-g.pl/im/16/27/10/z16936982V,kadr-z-filmu--The-room----na-nim-Tommy-Wiseau--pom.jpg)](https://wyborcza.pl/7,101707,22959130,tommy-wiseau-to-tomasz-wieczorkiewicz-najgorszy-rezyser-swiata.html)

The network should be able to generate a caption along the lines of:

> A fearless hero facing down a horde of his most terrifying enemies with his beloved football

# Backlog

## 22 - 28.04

Overall the objective of this week is to get familiar with Show&Tell code, refresh on PyTorch to become a true tensor-ninja and train a first captioning model

- [ ] Organise work:
    * [x] Basic README
    * [ ] Download MSCoco
    * [ ] Decide on PyTorch version
    * [ ] Get access to GPU for training
    * [ ] Pipfile (vs requirements.txt) (+ Docker ?)
    * [ ] How to include SemStyle code? git submodules vs fork vs shameless copy vs rewrite? What's their license?
    * [ ] (Unit) Tests? Tox? setup.py?
    * [ ] CLI, Click/python-fire?
    * [ ] Code Style
    * [ ] gitignore
- [ ] Make the code of Show&Tell runnable with recent PyTorch version
- [ ] Train the network on MSCoco
- [ ] Add GRU encoder


# References

## Primary

* [SemStyle](https://arxiv.org/abs/1805.07030)
* [Show & Tell](https://arxiv.org/pdf/1609.06647.pdf)
* [Show, Attend & Tell](https://arxiv.org/pdf/1502.03044.pdf)
  - [Along with PyTorch code](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

## Also inspired by

* http://openaccess.thecvf.com/content_cvpr_2017/papers/Gan_StyleNet_Generating_Attractive_CVPR_2017_paper.pdf
* http://openaccess.thecvf.com/content_ECCV_2018/papers/Tianlang_Chen_Factual_or_Emotional_ECCV_2018_paper.pdf
* http://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_MSCap_Multi-Style_Image_Captioning_With_Unpaired_Stylized_Text_CVPR_2019_paper.pdf


