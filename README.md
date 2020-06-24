# Unsupervised Shakespeare Style Captioning

Following a rapid development in state-of-the-art image capitoning models the goal of this master thesis is to train a neural network inspired by [SemStyle paper](https://arxiv.org/abs/1805.07030) to achieve image captioning in Shakespearian style. For instance given an image:

[![Tommy Wiseau](https://bi.im-g.pl/im/16/27/10/z16936982V,kadr-z-filmu--The-room----na-nim-Tommy-Wiseau--pom.jpg)](https://wyborcza.pl/7,101707,22959130,tommy-wiseau-to-tomasz-wieczorkiewicz-najgorszy-rezyser-swiata.html)

The network should be able to generate a caption along the lines of:

> A fearless hero facing down a horde of his most terrifying enemies with his beloved football

# Backlog

- [x] Organise work:
    * [x] Basic README
    * [x] Download MSCoco
    * [x] Decide on PyTorch version
    * [x] Get access to GPU for training
    * [x] Pipfile (vs requirements.txt) (+ Docker ?)
    * [x] How to include ShowAttend&Tell code? git submodules vs fork vs shameless copy vs rewrite? What's their license?
    * [x] (Unit) Tests? Tox? setup.py?
    * [x] CLI, Click/python-fire?
    * [x] Code Style
    * [x] gitignore
- [x] Make the code of Show&Tell runnable with recent PyTorch version
- [x] Train the network on MSCoco
- [ ] Evaluate the network on MSCoco and store results
- [x] Extract features from MSCoco images to hdf5 feature dataset
    * [x] Define the model
    * [x] Save the data
- [x] Encode the captions (Word Map)
- [ ] Train variation of *Term Generator* network from semstyle:
    * [ ] Define the model
    * [ ] Use fully connected + embedding as term extractor
    * [ ] Transform captions to terms (as in the paper)
      * [ ] Filtering non-semantic words
      * [ ] Lemmatization and tagging
      * [ ] Verb abstraction
- [ ] Replace the fully connected layers with RNN



## Comments


### 12.05 - 19.05

Fire is cool for quick development, but is limited - no config file support, weird documentation.
Migrate to click/milc at some point

What do I need in the dataset?
  * Features - h5py
  * Split - h5py
  * Captions - json


### 05.05 - 12.05

TODO


### 28.04 - 05.05

Keypoints:

- I've received access via ssh to a PC with a TITAN Xp GPU
- I had quite a few admin-ish problems with it (details below). I've solved them, updated the code to run with a recent python version. 
- One thing I noticed that striked me as odd (potentially a bug somewhere) is that using a larger batch-size resulted in lower memory usage as reported by `nvidia-smi`
- The code was mostly pytorch 1.5 compliant
- The network is set for 120 epochs which should take around 6 days, but it saves the best checkpoint after each epoch (and it's possible to resume training so I should have some sensible results in 1-2 days)



**Admin trouble**: TLDR; the wrong version of nvidia drivers was installed and it caused weird behaviour.
Long story:
The installed nvidia drivers were outdated and didn't work with the cuda version pytorch 1.5 ships with. So I had to `sudo apt update && sudo apt upgrade` them. This actually broke the drivers entirely. I've spent a lot of time googling for a solution that didn't involve messing with blacklisting nouveau drivers - I knew that if the machine fails to reboot I will be stuck waiting for somebody to physically turn the PC on. Finally I've figured out that this wasn't the latest major release of these drivers and I purged everything nvidia related and installed nvidia-drivers-440 instead of nvidia-drivers-418; in accordance to what was reported by `ubuntu-drivers devices` and the nvidia website. 



### 22 - 28.04

**ShowAttend&Tell code**: There are two problems

* Their code doesn't have a license (I can't reuse it freely). I've already bumped the relevant [issue](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/81) on github. For now I will probably have to assume they will release the project under a permissive license.
* How to use the code: FORK or rewrite. Rewrite would be better, fork is easier. For now fork seems a good idea, then maybe a rewrite once I understand all the nuances and focus on SemStyle

**Pytorch version**: I don't see any valid reasons to go with something else than latest stable torch (that is 1.5)

**Pipfile**: For now pipfile seems to be the most resonable solution. At some point it might be good to provide a dockerfile for easier deployment on different machines.

**Tests**: Testing machine learning code is not trivial. That's because it's hard to reason about models behaviour. There aren't any established industry standart testing methods, but [this article](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) should serve as a good inspiration. I will use [pytest](https://docs.pytest.org/en/latest/) as my testing framework and will also try to implement something like [torchtest](https://github.com/suriyadeepan/torchtest) (the library is simple & they use GPL-3 license & I want to use MIT license)

**Tox, setup.py**: I won't be writing a library so setup.py won't be needed. Tox is an overkill.

**Code style**:  I'm going to be using black auto-formatter as I like their coincise style.

**CLI**: I'm going with [Fire](https://github.com/google/python-fire)


## Lessons learned

  - Imports, Importlib, project structure:
    * Folders without `__init__.py` are namespaces not packages
    * Use relative imports
    * Run scripts using relative imports as `python -m`
    * Use `__main__.py` to be able to run the whole package as script
  - Logging
    * More configurable than printing
    * Gives the ability to naturally alter printing behaviour without altering codebase
    * Doesn't play well with things like tqdm
  - Fire (python-fire):
    * Good for prototyping
    * Bad for fine-grained control
  - Pipenv
    * Great idea, the future
    * Tool is unstable, unreliable and causes pain
  - Configuration
    * Configuring machine-learning from the command line sucks. There are just to many options to pass each time.
    * Config python file is better
    * Perhaps something like [Hydra](https://hydra.cc/docs/next/intro/) is the option for future

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


