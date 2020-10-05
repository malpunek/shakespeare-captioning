# Backlog

- [x] Organise work
    * [x] Basic README
    * [x] Download [MSCOCO](https://cocodataset.org)
    * [x] Decide on [PyTorch](https://pytorch.org) version
    * [x] Get access to GPU for training
    * [x] Pipfile (vs requirements.txt) (+ Docker ?)
    * [x] How to include ShowAttend&Tell code? git submodules vs fork vs shameless copy
      vs rewrite? What's their license?
    * [x] (Unit) Tests? Tox? setup.py?
    * [x] CLI, Click/python-fire?
    * [x] Code Style
    * [x] gitignore
- [x] Make the code of Show&Tell runnable with recent [PyTorch](https://pytorch.org) version
- [x] Train the network on [MSCOCO](https://cocodataset.org)
- [x] Evaluate the network on [MSCOCO](https://cocodataset.org) and store results
- [x] Extract features from [MSCOCO](https://cocodataset.org) images to hdf5 feature
  dataset
    * [x] Define the model
    * [x] Save the data
- [x] Preprocessing
    * [x] Convert to more convenient scheme
    * [x] Mapping verbs - [WordNet](https://wordnet.princeton.edu)
    * [x] Mapping verbs - [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/) -
      [SEMAFOR](https://github.com/Noahs-ARK/semafor)
    * [x] Mapping verbs - [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/) -
      [Open-SESAME](https://github.com/swabhs/open-sesame)
- [x] *Term Generator*
    * [x] Encoder
    * [x] Decoder (fully connected + embedding as decoder)
    * [x] Join networks
    * [x] torch dataset
    * [x] Training code
    * [x] Testing
    * [x] Decoder (GRU-based)
    * [x] Decoder Inference
    * [x] Evaluation (precision, recall)
    * [x] Train
- [x] *Language Generator*
    * [x] Encoder
    * [x] Decoder
    * [x] Decoder Attention
    * [x] torch datasets
    * [x] Testing
    * [x] Inference
    * [x] Train
- [x] Full network
    * [x] Joint word map for training
    * [x] Full image to caption network
    * [x] Full inference
    * [x] Evaluation
- [x] Experiments
    * [x] Try different CNN feature extractors
    * [x] Pick the best one
    * [x] Train on shakespeare - [WordNet](https://wordnet.princeton.edu)
    * [x] Train on shakespeare -
      [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/)
    * [x] Train on both shakespeare modern and original
    * [x] Train on Tolkien
    * [x] Train on Tolkien with Terms Merging
- [ ] K-beam search
- [ ] Intermediate conclusions
    * [ ] Quantify everything
    * [ ] Hypothesise and test
    * [ ] Publish results (so far)
    * [ ] Plan future experiments
- [ ] Polishing
    * [ ] Todos
    * [ ] Optimize datasets
    * [ ] Fix (remove) obsolete code
    * [ ] Cleanse requirements.txt
    * [ ] Installation guide


# Work log & Comments

## 1.09 - 8.09

I performed four different experiments this week, and performed a lot of qualitative
analysis. I started the outline for my the thesis.

## 23.06 - 1.09

These were crazy holidays and I didn't have much time to work on the project.

One notable addition was that after seeing the results of the LanguageGenerator I've
redone the whole preprocessing from scratch this time incorporating Open-sesame as the
frame annotating tool.

## 9.06 - 23.06 (two weeks)

Although I gained some RNN experience building the TermGenerator coding the
LanguageGenerator wasn't easier. One major problem that I had was that the decoder used
attention. But contrary to my expectations the attention was involved not in preparing
input for the GRUCell, but it added some input to the last fully connected layer. Also
the code from the original paper is provided in two forms (theano and [PyTorch](https://pytorch.org)) and they
do different things. So i had to follow the equation from the paper step by step to
determine what's the correct way of approaching the problem. At the end of the second
week I finally trained the first model. The results were faithfull to the image, but
they weren't styled in any way.

## 26.05 - 9.06 (two weeks)

Finally coding the model. The first part was easy - to build the TermGenerator from the
original paper but using Linear layers as decoder. As this was my first ever experience
with RNNs switching to GRU based decoder turned out to be a bit tricky. Especially that
I didn't just want to get the code done, but also to understand what's behind it. After
some struggling, couple of tutorials I got it right. Then came the time for the code of
training and inference. The RNNs run differently in training and inference (wiki.
teacher forcing) so that was an additional issue. I performed some experiments and after
quantitative analysis came up with the conclusion that using ResNet-101 as CNN feature
extractor gives better results than using Inceptionv3.

The model achieves 45.9% precision and 58.9% recall. That is significantly lower on
precision and higher on recall when comparing with results from the original paper.

## 19.05 - 26.05

Turns out mapping verbs into [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/)
frames is not trivial. In the original paper the authors used
[SEMAFOR](https://github.com/Noahs-ARK/semafor) to do the job. Unfortunately that tool
is abandoned and it's use is discouraged by it's authors. Instead they recommend
[Open-SESAME](https://github.com/swabhs/open-sesame). Unfortunately
[Open-SESAME](https://github.com/swabhs/open-sesame) is written in python 2 and as such
will become unusable after January 2021. That's why I decided to (for now) skip the
frame mapping part in whole. The idea was to get the network up and running as quickly
as possible and worry about preprocessing later. I used [SpaCy](https://spacy.io) for POS recognizing and
tagging.

## 12.05 - 19.05

Fire is cool for quick development, but is limited - no config file support, weird
documentation. I wanted to migrate to click, but I had some issues with relative imports
before i managed to do that. I decided to do the imports the right way. To solve the
problems i've watched an excellent [talk](https://youtu.be/0oTh1CXRaQ0) on imports and
importlib from [PyCon](https://pycon.org). The talk has also solved my cli issues - i
concluded to use built-in `python -m` for running diffrent commands and a config.py file
with a tiny bit of argparse for configuration.

How do I store data?
  * Features - h5py
  * Split - h5py
  * Captions - json

## 05.05 - 12.05

I had some deliveries for other projects so this week was out. The results from Show,
Attend and Tell model were decent.

TODO undig the actual results.

## 28.04 - 05.05

Keypoints:

- I've received access via ssh to a PC with a TITAN Xp GPU
- I had quite a few admin-ish problems with it (details below). I've solved them,
  updated the code to run with a recent python version.
- One thing I noticed that striked me as odd (potentially a bug somewhere) is that using
  a larger batch-size resulted in lower memory usage as reported by `nvidia-smi`
- The code was mostly [PyTorch](https://pytorch.org) 1.5 compliant
- The network is set for 120 epochs which should take around 6 days, but it saves the
  best checkpoint after each epoch (and it's possible to resume training so I should
  have some sensible results in 1-2 days)



**Admin trouble**: TLDR; the wrong version of nvidia drivers was installed and it caused
weird behaviour. Long story: The installed nvidia drivers were outdated and didn't work
with the cuda version [PyTorch](https://pytorch.org) 1.5 ships with. So I had to `sudo apt update && sudo apt
upgrade` them. This actually broke the drivers entirely. I've spent a lot of time
googling for a solution that didn't involve messing with blacklisting nouveau drivers -
I knew that if the machine fails to reboot I will be stuck waiting for somebody to
physically turn the PC on. Finally I've figured out that this wasn't the latest major
release of these drivers and I purged everything nvidia related and installed
nvidia-drivers-440 instead of nvidia-drivers-418; in accordance to what was reported by
`ubuntu-drivers devices` and the nvidia website.



## 22 - 28.04

**ShowAttend&Tell code**: There are two problems

* Their code doesn't have a license (I can't reuse it freely). I've already bumped the
  relevant
  [issue](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/81)
  on github. For now I will probably have to assume they will release the project under
  a permissive license.
* How to use the code: FORK or rewrite. Rewrite would be better, fork is easier. For now
  fork seems a good idea, then maybe a rewrite once I understand all the nuances and
  focus on SemStyle

**[PyTorch](https://pytorch.org) version**: I don't see any valid reasons to go with something else than latest
stable [PyTorch](https://pytorch.org) (that is 1.5)

**Pipfile**: For now pipfile seems to be the most resonable solution. At some point it
might be good to provide a dockerfile for easier deployment on different machines.

**Tests**: Testing machine learning code is not trivial. That's because it's hard to
reason about models behaviour. There aren't any established industry standart testing
methods, but [this
article](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765)
should serve as a good inspiration. I will use
[pytest](https://docs.pytest.org/en/latest/) as my testing framework and will also try
to implement something like [torchtest](https://github.com/suriyadeepan/torchtest) (the
library is simple & they use GPL-3 license & I want to use MIT license)

**Tox, setup.py**: I won't be writing a library so setup.py won't be needed. Tox is an
overkill.

**Code style**:  I'm going to be using black auto-formatter as I like their coincise
style.

**CLI**: I'm going with [Fire](https://github.com/google/python-fire)


# Lessons learned

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
    * Configuring machine-learning from the command line sucks. There are just to many
      options to pass each time.
    * Config python file is better
    * Perhaps something like [Hydra](https://hydra.cc/docs/next/intro/) is the option
      for future
  - Do NOT trust machine learning tools
    * Researchers are often poor developers
    * The nature of the job is that they usually write poor quality code
    * I should trust my judgment more
  - GET SHIT DONE
    * And polish code later
    * Jupyter notebooks are awesome
    * There is a hack for jupyter in vscode with package structure (below)
    * Writing good, optimal code takes way longer than writing shit, brute-force code
      and waiting for it to finish execution
    * Scripts that are executed once can be written poorly, others that are rerunned
      many times can (and probably should) be more polished
  - Running
    * Use git tag to distinguish between runs
    * Running on small subset of the data is a good test


Jupyter hack:
```python
try:
    __IPYTHON__
    import sys

    sys.path.insert(0, '<absolute_path_to_folder>')
    __package__ = "package.subpackage"

    sys.argv = ["nothing.py", "-t", "25", "-x", "999"]  # arguments for argparse

except NameError:
    pass
```


# All links

* [MSCOCO](https://cocodataset.org)
* [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/)
* [WordNet](https://wordnet.princeton.edu)
* [SpaCy](https://spacy.io)
* [SEMAFOR](https://github.com/Noahs-ARK/semafor)
* [Open-SESAME](https://github.com/swabhs/open-sesame)
* [Hydra](https://hydra.cc/docs/next/intro/)
* [Fire](https://github.com/google/python-fire)
* [pytest](https://docs.pytest.org/en/latest/)
* [torchtest](https://github.com/suriyadeepan/torchtest)
* [PyCon](https://pycon.org)
* [Pycon talk on imports](https://youtu.be/0oTh1CXRaQ0)
* [Article on testing ML
  code](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765)
