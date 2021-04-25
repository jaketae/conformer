# Conformer

PyTorch implementation of [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al., 2020).

## Quickstart

Clone this repository.

```
git clone https://github.com/jaketae/conformer.git
```

Navigate to the cloned directory. You can start using the model via

```python
>>> from conformer import ConformerEncoder
>>> model = ConformerEncoder()
```

By default, the model comes with the following parameters:

```python
ConformerEncoder(
    num_blocks=6,
    d_model=256,
    num_heads=4,
    max_len=512,
    expansion_factor=4,
    kernel_size=31,
    dropout=0.1,
)
```

## Introduction

The [Transformer](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) has proven to be immensely successful in various domains, such as machine translation, [language modeling](https://arxiv.org/abs/1810.04805), and more recently, [computer vision](https://arxiv.org/abs/2010.11929). An important reason behind the success of the transformer architecture is self-attention, which allows the model to attend to the entire input sequence to generate rich feature representations.

A more traditional model architecture, convolution neural networks have widely used in the vision domain. The sliding kernel structure encodes meaningful inductive biases such as translation invariance, making them suitable as local feature extractors.

The Conformer seeks to combine the best of both worlds: global features are extracted by the transformer, whereas local features are learned by the convolution module. The Conformer model has proven to be effective in [automatic speech recognition](https://paperswithcode.com/task/speech-recognition).

## Note

While the original paper used Conformer specifically in the context of ASR, I implemented this model in the hopes of applying it as a general feature encoder in audio generation tasks, such as speech prosody transfer, voice conversation, and singing voice synthesis. For this reason, the current implementation only includes the encoder portion of the Conformer architecture and also lacks components such as downsampling and SpecAugment.

## Credit

This implementation was heavily influenced by [Soohwan Kim's implementation of Conformer](https://github.com/sooftware/conformer). The skewing logic employed in relative positional encoding was inspired by [Prayag Chatha's implementation](https://github.com/chathasphere/pno-ai) of [Music Transformer](https://arxiv.org/abs/1809.04281) (Huang et al., 2018).
