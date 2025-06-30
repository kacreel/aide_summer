---
title: "What exactly are keys, queries, and values in attention mechanisms?"
source: "https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms"
author:
  - "[[SeanSean                    4]]"
  - "[[28744 gold badges1818 silver badges3939 bronze badges]]"
  - "[[dontloodontloo                    17k99 gold badges6363 silver badges8989 bronze badges]]"
  - "[[Sam TsengSam Tseng                    94166 silver badges55 bronze badges]]"
  - "[[monmon                    1]]"
  - "[[7681313 silver badges2121 bronze badges]]"
  - "[[NitinNitin                    37911 gold badge33 silver badges99 bronze badges]]"
  - "[[EmilEmil                    37122 silver badges77 bronze badges]]"
  - "[[Sean KernitsmanSean Kernitsman                    15111 silver badge33 bronze badges]]"
  - "[[Sergey SkrebnevSergey Skrebnev                    32133 silver badges33 bronze badges]]"
published: 2019-08-13
created: 2025-06-27
description: "How should one understand the keys, queries, and values that are often mentioned in attention mechanisms?I've tried searching online, but all the resources I find only speak of them as if the reader"
tags:
---
Asked

Modified [1 year, 4 months ago](https://stats.stackexchange.com/questions/421935/?lastactivity "2024-02-26 06:13:43Z")

Viewed 294k times

How should one understand the keys, queries, and values that are often mentioned in attention mechanisms?

I've tried searching online, but all the resources I find only speak of them as if the reader already knows what they are.

Judging by the paper written by Bahdanau (*[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)*), it seems as though values are the annotation vector $h$ but it's not clear as to what is meant by "query" and "key."

The paper that I mentioned states that attention is calculated by

$$
c_i = \sum^{T_x}_{j = 1} \alpha_{ij} h_j
$$

with

$$
\begin{align}
\alpha_{ij} & = \frac{e^{e_{ij}}}{\sum^{T_x}_{k = 1} e^{ik}} \\\\
e_{ij} & = a(s_{i - 1}, h_j)
\end{align}
$$

Where are people getting the key, query, and value from these equations?

Thank you.

13

The key/value/query formulation of attention is from the paper [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).

> How should one understand the queries, keys, and values

The key/value/query concept is analogous to retrieval systems. For example, when you search for videos on Youtube, the search engine will map your **query** (text in the search bar) against a set of **keys** (video title, description, etc.) associated with candidate videos in their database, then present you the best matched videos (**values**).

The attention operation can be thought of as a retrieval process as well.

As mentioned in the paper you referenced ([Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)), attention by definition is just a weighted average of values,

$$
c=\sum_{j}\alpha_jh_j
$$
 where $\sum \alpha_j=1$.

If we restrict $\alpha$ to be a one-hot vector, this operation becomes the same as retrieving from a set of elements $h$ with index $\alpha$. With the restriction removed, the attention operation can be thought of as doing "proportional retrieval" according to the probability vector $\alpha$.

It should be clear that $h$ in this context is the **value**. The difference between the two papers lies in how the probability vector $\alpha$ is calculated. The first paper (Bahdanau et al. 2015) computes the score through a neural network 
$$
e_{ij}=a(s_i,h_j), \qquad \alpha_{i,j}=\frac{\exp(e_{ij})}{\sum_k\exp(e_{ik})}
$$
 where $h_j$ is from the encoder sequence, and $s_i$ is from the decoder sequence. One problem of this approach is, say the encoder sequence is of length $m$ and the decoding sequence is of length $n$, we have to go through the network $m*n$ times to acquire all the attention scores $e_{ij}$.

A more efficient model would be to first project $s$ and $h$ onto a common space, then choose a similarity measure (e.g. dot product) as the attention score, like 
$$
e_{ij}=f(s_i)g(h_j)^T
$$
 so we only have to compute $g(h_j)$ $m$ times and $f(s_i)$ $n$ times to get the projection vectors and $e_{ij}$ can be computed efficiently by matrix multiplication.

This is essentially the approach proposed by the second paper (Vaswani et al. 2017), where the two projection vectors are called **query** (for decoder) and **key** (for encoder), which is well aligned with the concepts in retrieval systems. (There are later techniques to further reduce the computational complexity, for example [Reformer](https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html), [Linformer](https://arxiv.org/pdf/2006.04768.pdf), [FlashAttention](https://arxiv.org/abs/2307.08691).)

> How are the queries, keys, and values obtained

The proposed multihead attention alone doesn't say much about how the queries, keys, and values are obtained, they can come from different sources depending on the application scenario.

> $$
> \begin{align}\text{MultiHead($Q$, $K$, $V$)} & = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^{O} \\
> \text{where head$_i$} &  = \text{Attention($QW_i^Q$, $KW_i^K$, $VW_i^V$)}
> \end{align}
> $$
>  Where the projections are parameter matrices:
> $$
> \begin{align}
> W_i^Q & \in \mathbb{R}^{d_\text{model} \times d_k}, \\
> W_i^K & \in \mathbb{R}^{d_\text{model} \times d_k}, \\
> W_i^V & \in \mathbb{R}^{d_\text{model} \times d_v}, \\
> W_i^O & \in \mathbb{R}^{hd_v \times d_{\text{model}}}.
> \end{align}
> $$

For unsupervised language model training like [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), $Q, K, V$ are usually from the same source, so such operation is also called self-attention.

For the machine translation task in the second paper, it first applies self-attention separately to source and target sequences, then on top of that it applies another attention where $Q$ is from the target sequence and $K, V$ are from the source sequence.

For recommendation systems, $Q$ can be from the target items, $K, V$ can be from the user profile and history.

13

I was also puzzled by the keys, queries, and values in the attention mechanisms for a while. After searching on the Web and digesting relevant information, I have a clear picture about how the keys, queries, and values work and why they would work!

Let's see how they work, followed by why they work.

### Attention to replace context vector

In a seq2seq model, we encode the input sequence to a **context vector**, and then feed this context vector to the decoder to yield expected good output.

However, if the input sequence becomes long, relying on only one context vector become less effective. We need all the information from the hidden states in the input sequence (encoder) for better decoding (the attention mechanism).

One way to utilize the input hidden states is shown below:[![Image source: https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3](https://i.sstatic.net/13ADZ.png)](https://i.sstatic.net/13ADZ.png) [Image source](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)

In other words, **in this attention mechanism, the context vector** `is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key` (this is a slightly modified sentence from [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)).

Here, the query is from the decoder hidden state, the key and value are from the encoder hidden states (key and value are the same in this figure). The score is the compatibility between the query and key, which can be a dot product between the query and key (or other form of compatibility). The scores then go through the softmax function to yield a set of weights whose sum equals 1. Each weight multiplies its corresponding values to yield the context vector which utilizes all the input hidden states.

Note that if we manually set the weight of the last input to 1 and all its precedences to 0s, we reduce the attention mechanism to the original seq2seq context vector mechanism. That is, there is no attention to the earlier input encoder states.

### Self-Attention uses Q, K, V all from the input

Now, let's consider the self-attention mechanism as shown in the figure below:

[![enter image description here](https://i.sstatic.net/J45g2.png)](https://i.sstatic.net/J45g2.png) [Image source](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)

The difference from the above figure is that the queries, keys, and values are **transformations** of the corresponding input state vectors. The others remain the same.

Note that we could still use the original encoder state vectors as the queries, keys, and values. So, **why we need the transformation**? The transformation is simply a matrix multiplication like this:

Query = I x W(Q)

Key = I x W(K)

Value = I x W(V)

where I is the input (encoder) state vector, and W(Q), W(K), and W(V) are the corresponding matrices to transform the I vector into the Query, Key, Value vectors.

What are the benefits of this matrix multiplication (vector transformation)?

The obvious reason is that if we do not transform the input vectors, the dot product for computing the weight for each input's value will always yield a maximum weight score for the individual input token itself. In other words, when we compute the n attention weights (j for j=1, 2,..., n) for input token at position i, the weight at i (j==i) is always the largest than the other weights at j=1, 2,..., n (j<>i). This may not be the desired case. For example, for the pronoun token, we need it to attend to its referent, not the pronoun token itself.

Another less obvious but important reason is that the **transformation may yield better representations for Query, Key, and Value**. Recall the effect of Singular Value Decomposition (SVD) like that in the following figure:

[![Application of SVD](https://i.sstatic.net/zPiHH.png)](https://i.sstatic.net/zPiHH.png)

[Image source](https://youtu.be/K38wVcdNuFc?t=10)

By multiplying an input vector with a matrix V (from the SVD), we obtain a better representation for computing the compatibility between two vectors, if these two vectors are similar in the topic space as shown in the example in the figure.

And these matrices for transformation can be learned in a neural network!

In short, by multiplying the input vector with a matrix, we got:

1. increase of the possibility for each input token to attend to other tokens in the input sequence, instead of individual token itself
2. possibly better (latent) representations of the input vector
3. conversion of the input vector into a space with a desired dimension, say, from dimension 5 to 2, or from n to m, etc (which is practically useful)

I hope this help you understand the queries, keys, and values in the (self-)attention mechanism of deep neural networks.

6

## Big picture

Basically Transformer builds a graph network where a node is a position-encoded token in a sequence.

During training:

1. Get un-connected tokens as a sequence (e.g. sentence).
2. Wires connections among tokens by having looked at the co-occurrences of them in billions of sequences.

What roles `Q` and `K` will play to build this graph network? You could be `Q` in your society trying to build the social graph network with other people. Each person in the people is `K` and you will build the connections with them. Eventually by having billions of interactions with other people, the connections become dependent on the contexts even with the same person `K`.

You may be superior to a person K at work, but K may be a master of martial art for you. As you remember such connections/relations with others based on the contexts, Transformer model (trained on a specific dataset) figures out such context dependent connections from Q to K (or from you to other person(s)), which is a **memory** that it offers.

If the layers go up higher, your individual identity as K will be blended into larger parts via going through the BoW process which plays the role.

With regard to the Markov Chain (MC), there is only one static connection from Q to K as `P(K|Q)` in MC as MC does not have the context **memory** that Transformer model offers.

## First, understand Q and K

First, focus on **the objective of `First MatMul`** in the [Scaled dot product attention](https://www.tensorflow.org/text/tutorials/transformer#scaled_dot_product_attention) using `Q` and `K`.

[![enter image description here](https://i.sstatic.net/MJIyF.png)](https://i.sstatic.net/MJIyF.png)

## Intuition on what is Attention

For the sentence **"jane** visits africa".

When your eyes see ***jane***, your brain looks for **the most related word** in the rest of the sentence to understand what **jane** is about (query). Your brain focuses or attends to the word **visit** (key).

This process happens for each word in the sentence as your eyes progress through the sentence.

## First MatMul as Inquiry System using Vector Similarity

The first `MatMul` implements an inquiry system or question-answer system that imitates this brain function, using Vector Similarity Calculation. Watch [CS480/680 Lecture 19: Attention and Transformer Networks](https://youtu.be/OyFJWRnt_AY?t=704) by professor Pascal Poupart to understand further.

> Think about the attention essentially being some form of approximation of SELECT that you would do in the database.  
> [![enter image description here](https://i.sstatic.net/nVvt9m.png)](https://i.sstatic.net/nVvt9m.png)

[![enter image description here](https://i.sstatic.net/bgwSb.png)](https://i.sstatic.net/bgwSb.png)

Think of the MatMul as an inquiry system that processes the inquiry: "For the word **`q`** that your eyes see in the given sentence, what is the most related word **`k`** in the sentence to understand what **`q`** is about?" The inquiry system provides the answer as the probability.

| q | k | probability |
| --- | --- | --- |
| jane | visit | 0.94 |
| visit | africa | 0.86 |
| africa | visit | 0.76 |

Note that the softmax is used to normalize values into probabilities so that their sum becomes 1.0.

[![enter image description here](https://i.sstatic.net/rQhuQ.jpg)](https://i.sstatic.net/rQhuQ.jpg)

There are multiple ways to calculate the similarity between vectors such as cosine similarity. Transformer attention uses simple **dot product**.

## Where are Q and K from

The transformer encoder training builds the weight parameter matrices `WQ` and `Wk` in the way `Q` and `K` builds the Inquiry System that answers the inquiry " **What is `k` for the word `q`** ".

The calculation goes like below where `x` is a sequence of position-encoded word embedding vectors that represents an input sentence.

1. Picks up a word vector (position encoded) from the input sentence sequence, and transfer it to a vector space **Q**. This becomes the **q** uery.  
	$Q = X \cdot W_{Q}^T$
2. Pick all the words in the sentence and transfer them to the vector space **K**. They become keys and each of them is used as **k** ey.  
	$K = X \cdot W_K^T$
3. For each (**q**, **k**) pair, their relation strength is calculated using dot product.  
	$q\_to\_k\_similarity\_scores = matmul(Q, K^T)$
4. Weight matrices $W_Q$ and $W_K$ are trained via the back propagations during the Transformer training.

We first needs to understand this part that involves **Q** and **K** before moving to ***V***.

[![enter image description here](https://i.sstatic.net/DWNTr.jpg)](https://i.sstatic.net/DWNTr.jpg)

Borrowing the code from [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy.

```
# B: Batch size
# T: Sequence length or max token size e.g. 512 for BERT. 'T' because of 'Time steps = Sequence length'
# D: Dimensions of the model embedding vector, which is d_model in the paper.
# H or h: Number of multi attention heads in Multi-head attention

def calculate_dot_product_similarities(
        query: Tensor,
        key: Tensor,
) -> Tensor:
    """
    Calculate similarity scores between queries and keys using dot product.

    Args:
        query: embedding vector of query of shape (B, h, T, d_k)
        key: embedding vector of key of shape (B, h, T, d_k)

    Returns: Similarities (closeness) between q and k of shape (B, h, T, T) where
        last (T, T) represents relations between all query elements in T sequence
        against all key elements in T sequence. If T is people in an organization,
        (T,T) represents all (cartesian product) social connections among them.
        The relation considers d_k number of features.
    """
    # --------------------------------------------------------------------------------
    # Relationship between k and q as the first MatMul using dot product similarity:
    # (B, h, T, d_k) @ (B, hH, d_k, T) ---> (B, h, T, T)
    # --------------------------------------------------------------------------------
    similarities = query @ key.transpose(-2, -1)            # dot product
    return similarities                                     # shape:(B, h, T, T)
```

## Then, understand how V is created using Q and K

## Second Matmul

Self Attention then generates the embedding vector called **attention value** as a bag of words (BoW) where each word contributes proportionally according to its relationship strength to **q**. This occurs for each **q** from the sentence sequence. The embedding vector is encoding the relations from **q** to all the words in the sentence.

Citing the [words](https://youtu.be/kCc8FmEb1nY?t=2625) from Andrej Karpathy:

> What is the easiest way for tokens to communicate. The easiest way is just average.

He makes it simple for the sake of tutorial but the essence is BoW.

[![enter image description here](https://i.sstatic.net/TBpsF.png)](https://i.sstatic.net/TBpsF.png)

```
def calculate_attention_values(
        similarities,
        values
):
    """
    For every q element, create a Bag of Words that encodes the relationships with
    other elements (including itself) in T, using (q,k) relationship value as the
    strength of the relationships.

    Citation:
    > On each of these projected versions of queries, keys and values we then perform
    > the attention function in parallel, yielding d_v-dimensional output values.

    \`\`\`
    bows = []
    for row in similarities:                    # similarity matrix of shape (T,T)
        bow = sum([                             # bow:shape(d_v,)
            # each column in row is (q,k) similarity score s
            s*v for (s,v) in zip(row,values)    # k:shape(), v:shape(d_v,)
=        ])
        bows.append(bow)                        # bows:shape(T,d_v)
    \`\`\`

    Args:
        similarities: q to k relationship strength matrix of shape (B, h, T, T)
        values: elements of sequence with length T of shape (B, h, T, d_v)

    Returns: Bag of Words for every q element of shape (B, h, T, d_v)
    """
    return similarities @ values     # (B,h,T,T) @ (B,h,T,d_v) -> (B,h,T,d_v)
```

## References

There are multiple concepts that will help understand how the self attention in transformer works, e.g. embedding to group similars in a vector space, data retrieval to answer query Q using the neural network and vector similarity.

- [CS25 I Stanford Seminar - Transformers United 2023: Introduction to Transformers w/ Andrej Karpathy](https://youtu.be/XfpMkf4rD6E?t=1395): Andrej Karpathy explained by regarding a sentence as a graph.
- [Transformers Explained Visually (Part 2): How it works, step-by-step](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34) give in-detail explanation of what the Transformer is doing.
- [CS480/680 Lecture 19: Attention and Transformer Networks](https://www.youtube.com/watch?v=OyFJWRnt_AY) - This is probably the best explanation I found that actually explains the attention mechanism from the database perspective.
- [Illustrated Guide to Transformers Neural Network: A step by step explanation](https://www.youtube.com/watch?v=4Bdc55j80l8)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) - It helps understand how word2vec works to group/categorize words in a vector space by pulling similar words together, and pushing away non-similar words using negative sampling.
- [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) - Continuation to understand embedding to pull together siimilars and pushing away non-similars in a vector space.
- [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer) - TensorFlow implementation of transformer
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - PyTorch implementation of Transformer

5

I'm going to try provide an English text example. The following is based solely on my intuitive understanding of the paper 'Attention is all you need'.

Say you have a sentence:

- I like Natural Language Processing, a lot!

Assume that we already have input word vectors for all the 9 tokens in the previous sentence. So, 9 input word vectors.

Looking at the encoder from the paper 'Attention is all you need', the encoder needs to produce 9 output vectors, one for each word. This is done, through the Scaled Dot-Product Attention mechanism, coupled with the Multi-Head Attention mechanism. I'm going to focus only on an intuitive understanding of the Scaled Dot-Product Attention mechanism, and I'm not going to go into the scaling mechanism.

Walking through an example for the first word 'I':

- The **query** is the input word vector for the token "I"
- The **keys** are the input word vectors for all the other tokens, and for the query token too, i.e (semi-colon delimited in the list below):
	\[like;Natural;Language;Processing;,;a;lot;!\] + \[I\]
- The word vector of the **query** is then DotProduct-ed with the word vectors of *each* of the **keys**, to get 9 scalars / numbers a.k.a "weights"
- These weights are then scaled, but this is not important to understand the intuition
- The weights then go through a 'softmax' which is a particular way of normalizing the 9 weights to values between 0 and 1. This becomes important to get a "weighted-average" of the **value** vectors, which we see in the next step.
- Finally, the initial 9 input word vectors a.k.a **values** are summed in a "weighted average", with the normalized weights of the previous step. This final step results in a **single output word vector representation** of the word "I"

Now that we have the process for the word "I", rinse and repeat to get word vectors for the remaining 8 tokens. We now have 9 output word vectors, each put through the Scaled Dot-Product attention mechanism. You can then add a new attention layer/mechanism to the encoder, by taking these 9 new outputs (a.k.a "hidden vectors"), and considering these as inputs to the new attention layer, which outputs 9 new word vectors of its own. And so on ad infinitum.

If this Scaled Dot-Product Attention layer summarizable, I would summarize it by pointing out that **each token (query) is free to take as much information using the dot-product mechanism from the other words (values), and it can pay as much or as little attention to the other words as it likes by weighting the other words with (keys)**. The real power of the attention layer / transformer comes from the fact that each token is looking at all the other tokens at the same time (unlike an RNN / LSTM which is restricted to looking at the tokens to the left)

The Multi-head Attention mechanism in my understanding is this same process happening independently in parallel a given number of times (i.e number of heads), and then the result of each parallel process is combined and processed later on using math. I didn't fully understand the rationale of having the same thing done multiple times in parallel before combining, but i wonder if its something to do with, as the authors might mention, the fact that each parallel process takes place in a separate Linear Algebraic 'space' so combining the results from multiple 'spaces' might be a good and robust thing (though the math to prove that is way beyond my understanding...)

5

See [Attention is all you need - masterclass](https://youtu.be/rBCqOTEfxvg?t=946), from 15:46 onwards Lukasz Kaiser explains what *q, K* and *V* are.

So basically:

- *q* = the vector representing a word
- *K* and *V* = your memory, thus all the words that have been generated before. Note that *K* and *V* can be the same (but don't have to).

So what you do with attention is that you take your current query (word in most cases) and look in your memory for similar keys. To come up with a distribution of relevant words, the softmax function is then used.

5

Tensorflow and Keras just expanded on their documentation for the Attention and AdditiveAttention layers. Here is a sneaky peek from the docs:

> The meaning of query, value and key depend on the application. In the case of text similarity, for example, query is the sequence embeddings of the first piece of text and value is the sequence embeddings of the second piece of text. key is usually the same tensor as value.

But for my own explanation, different attention layers try to accomplish the same task with mapping a function $f: \Bbb{R}^{T\times D} \mapsto \Bbb{R}^{T \times D}$ where T is the hidden sequence length and D is the feature vector size. For the case of global self- attention which is the most common application, you first need sequence data in the shape of $B\times T \times D$, where $B$ is the batch size. Each forward propagation (particularly after an encoder such as a Bi-LSTM, GRU or LSTM layer with `return_state and return_sequences=True` for TF), it tries to map the selected hidden state (Query) to the most similar other hidden states (Keys). After repeating it for each hidden state, and `softmax` the results, multiply with the keys again (which are also the values) to get the vector that indicates how much attention you should give for each hidden state. I hope this helps anyone as it took me days to figure it out.

[![The flow of any attention layer](https://i.sstatic.net/SG66z.png)](https://i.sstatic.net/SG66z.png)

- **Q** ueries is a set of vectors you want to calculate attention for.
- **K** eys is a set of vectors you want to calculate attention against.
- As a result of dot product multiplication you'll get set of weights **a** (also vectors) showing how attended each query against **K** eys. Then you multiply it by **V** alues to get resulting set of vectors.

[![enter image description here](https://i.sstatic.net/v7Jyi.png)](https://i.sstatic.net/v7Jyi.png)

Now let's look at word processing from the article "Attention is all you need". There are two self-attending (xN times each) blocks, separately for inputs and outputs plus cross-attending block transmitting knowledge from inputs to outputs.

Each self-attending block gets just one set of vectors (embeddings added to positional values). In this case you are calculating attention for vectors against each other. So **Q=K=V**. You just need to calculate attention for each **q** in **Q**.

Cross-attending block transmits knowledge from inputs to outputs. In this case you get **K=V** from inputs and **Q** are received from outputs. I think it's pretty logical: you have database of knowledge you derive from the inputs and by asking **Q** ueries from the output you extract required knowledge.

*How attention works: dot product between vectors gets bigger value when vectors are better aligned. Then you divide by some value (scale) to evade problem of small gradients and calculate softmax (when sum of weights=1). At this point you get set of weights sum=1 that tell you for which vectors in **K** eys your **q** uery is better aligned. All that's left is to multiply by **V** alues.*

4

> Where are people getting the key, query, and value from these equations?

[The paper you refer to](https://arxiv.org/abs/1409.0473) *does not* use such terminology as "key", "query", or "value", so it is not clear what you mean in here. There is no single definition of "attention" for neural networks, so my guess is that you confused two definitions from different papers.

In the paper, the attention module has weights $\alpha$ and the values to be weighted $h$, where the weights are derived from the recurrent neural network outputs, as described by the equations you quoted, and on the figure from the paper reproduced below.

[![![enter image description here](https://i.sstatic.net/0dzbO.png)](https://i.sstatic.net/0dzbO.png)

Similar thing happens in the Transformer model from the [*Attention is all you need* paper by Vaswani et al](https://arxiv.org/abs/1706.03762), where they do use "keys", "querys", and "values" ($Q$, $K$, $V$). Vaswani et al define the attention cell [differently](https://stats.stackexchange.com/questions/387114/on-masked-multi-head-attention-and-layer-normalization-in-transformer-model/387138#387138):

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\Big(\frac{QK^T}{\sqrt{d_k}}\Big)V
$$

What they also use is multi-head attention, where instead of a single value for each $Q$, $K$, $V$, they provide multiple such values.

[![enter image description here](https://i.sstatic.net/jWduk.png)](https://i.sstatic.net/jWduk.png)

Where in the Transformer model, the $Q$, $K$, $V$ values can either come from the same inputs in the encoder (bottom part of the figure below), or from different sources in the decoder (upper right part of the figure). This part is crucial for using this model in translation tasks.

[![enter image description here](https://i.sstatic.net/DEmfr.png)](https://i.sstatic.net/DEmfr.png)

In both papers, as described, the values that come as input to the attention layers *are calculated from the outputs of the preceding layers* of the network. Both paper define different ways of obtaining those values, since they use different definition of attention layer.

This is an add up of what is K and V and why the author use different parameter to represent K and V. Short answer is technically K and V can be different and there is a case where people use different values for K and V.

## K and V can be different! Example Offered!

**What are K and V? Are they the same?**

The short answer is that they can be the same, but technically they do not need to be the same.

**Briefly introduce K, V, Q but highly recommend the previous answers**: In the [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) paper, this Q, K, V are first introduced. In that paper, generally(which means not self attention), the Q is the decoder embedding vector(the side we want), K is the encoder embedding vector(the side we are given), V is also the encoder embedding vector. And this attention mechanism is all about trying to find the relationship(weights) between the Q with all those Ks, then we can use these weights(freshly computed for each Q) to compute a new vector using Vs(which should related with Ks). If this is self attention: Q, V, K can even come from the same side -- eg. compute the relationship among the features in the encoding side between each other.(Why not show strong relation between itself? Projection.)

**Case where they are the same**: here in the Attention is all you need paper, they are the same before projection. Also in this [transformer code tutorial](https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb), V and K is also the same before projection.

**Case where K and V is not the same**: In the paper [End-to-End Object Detection](https://arxiv.org/pdf/2005.12872.pdf) Appendix A.1 Single head(this part is an introduction for multi head attention, you do not have to read the paper to figure out what this is about), they offer an intro to multi-head attention that is used in the Attention is All You Need papar, here they add some positional info to the K but not to the V in equation (7), which makes the K and the V here are not the same.

Hope this helps.

---

Edit: As recommended by @alelom, I put my **very shallow and informal understand of K, Q, V** here.

For me, informally, **the Key, Value and Query are all features/embeddings**. Though it actually depends on the implementation but commonly, **Query** is feature/embedding from the output side(eg. target language in translation). **Key** is feature/embedding from the input side(eg. source language in translation), and for **Value**, basing on what I read by far, it should certainly relate to / be derived from Key since the parameter in front of it is computed basing on relationship between K and Q, but it can be a feature that is based on K but being added some external information or being removed some information from the source(like some feature that is special for source but not helpful for the target)...

What I have read(very limited, and I cannot recall the complete list since it is already a year ago, but all these are the ones that I found helpful and impressive, and basically it is just a summary of what I referred above...):

- Neural Machine Translation By Jointly Learning To Align And Translate.
- Attention Is All You Need. and a tensorflow tutorial of transformer:[https://www.tensorflow.org/text/tutorials/nmt\_with\_attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention).
- End-to-end object detection with Transformers, and its code: [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr).
- lil'log: [https://lilianweng.github.io/posts/2018-06-24-attention/](https://lilianweng.github.io/posts/2018-06-24-attention/)

5

This is a concept that is the heart of the Transformer architecture and is difficult to explain or grasp.

We can say that in the process of learning the correct 'next' token to predict, three sets of weights per token are learned by backpropagating the loss - the **Key**, the **Query** and the **Value** weights. These form the base of the "Attention" mechanism. The video beautifully explains this at this [location](https://www.youtube.com/watch?t=1460&v=g2BRIuln4uc&feature=youtu.be).

The [concept of Vector dot produc](https://stats.stackexchange.com/a/452359/191675) t is used to calculate the Value Vectors, which is the sum of the contribution of the dot product of Query and Key vectors. The intuition is that similar vectors in the Vector embedding space will have a larger dot product value and higher contribution. The Weights are then adjusted via Backpropagation which means that the learned weights represent a better Contextual Vector Embedding space for each token. ( Key and Query weights are multi-dimensional and there are multiple attention heads, so it is not one vector space but many)

In Transformers, there are multiple attention heads, so, which attention head to weightage more can also be tuned via weights. Basically Transformer network is this architecture, where the intuition of causal relationship between tokens is encoded as learnable weights in linear neural net layers.

I have tried to explain this in a simplified image below. It may not be fully accurate, but the overall high-level mechanism is right

[![Approximate working of a single attention head in Transformer Network](https://i.sstatic.net/gkcHf.png)](https://i.sstatic.net/gkcHf.png)

3

A lot of these posts are super technical, but I think make grasping the general idea difficult. So, **I'm going to give a non-technical explanation.** (If you want a technical explanation, [this piece](https://medium.com/@london-lowmanstone/a-simpler-description-of-the-attention-mechanism-573e9e7f42b0) dives into more of the complexities.)

I'm going to use an analogy with dating apps. You have...

1. A man who is trying to figure out how to improve his dating profile. (This man is the *query*.)
2. A woman on the app. (This woman is the *key*.)
3. That woman's ideal partner (someone she would want to match with immediately). (Her ideal partner is the *value*.)

Now, note that a dating profile is a *representation* of you. So, the man's goal is to improve his own representation.

The *attention* mechanism is a special app that can automatically improve the man's profile by taking into account who he wants to match with and turning his profile into a fusion of their ideal partners. The attention mechanism app works by learning to do the following:

1. Given a particular man (the *query*), generate a new profile for that man that will be used to find women he might want to match with. (This function is called *applying query weights*.)
2. Given a particular woman (the *key*), generate a new profile for that woman that will be used to find men that might want to match with her. (This function is called *applying key weights*.)
3. Given the key woman's ideal partner (the *value*), generate a profile for that ideal partner. (This function is called *applying value weights*.)

Now, the attention mechanism app doesn't learn how to perform these tasks by being given the best-performing human-designed profiles and learning how to copy or imitate them. Instead, men will use the profiles from the app, and the results of whether they get more matches or not will help determine small tweaks the app can make to do better next time. (This is *gradient descent*.)

Given that these are the three things the attention app can improve at, here's how the app goes about coming up with suggestions for an improved profile for a given man:

- Step 1. Generate a new profile for the man that will be used to find women he might want to match with. (Apply the *query weights* to the *query*.)
- Step 2. For every woman on the app, generate a profile for them that will be used to find men that might want to match with her. (Apply the *key weights* to each *key*.)
- Step 3. Compare the man's generated profile to each of the women's generated profiles. We assume that the more similar they are to each other, the more the man likely wants to match with her, so it's really important that step 1 and 2 are done well. (Generate *attention scores*.)
- Step 4. Given the ideal partner for every woman on the app, generate a profile for each of the ideal partners. (Apply the *value weights* to each *value* to get a *new value representation*.)
- Step 5. Take all the ideal profiles from step 4 and combine them into a new final profile suggestion for the man, but weight how much each woman's ideal partner profile influences the man's final suggested profile based on how likely it is that the man wants to match with her. (Output a *weighted sum* of the *new value representations* based on the *attention scores*.)

And that's it! That's how the attention mechanism works.

Now, you may be thinking "this is a great analogy, but how is attention applied in practice?" Great question! I come from a natural language processing background, so I'll explain how it's used in my field.

Those of us in natural language processing work with text, and so for us, queries, keys, and values are all words (technically *tokens*). We use attention to come up with better representations for the query words based on the key/value words. Often this is used in translation systems where the (query) word in the translated language may be a translation of multiple (key/value) words in the input language. The process of creating male and female profiles with similarities to see who the man wants to appeal to is analogous to creating new query and key vectors via the learned weights to represent the query and key/value words such that vectors pointing in the same direction indicate that the translation query word should be heavily impacted by that input key/value word. Then, just like how the suggested male profile was a weighted sum of all the generated ideal profiles, the representation of the translated query word becomes a weighted sum of all of the generated (via the learned value weight matrix) value vectors of the key/value words.

Now, you may also be wondering "What's the difference between attention and self-attention?" Notice how men and women were split up in this example? It was only men (*decoder tokens*) who were trying to improve their profiles, and they were only trying to improve them for women (*encoder tokens*). Self-attention is when everyone is trying to improve their profiles for everyone (better representations for all *input tokens*)! It's the same technique, but now everyone is a key and everyone's ideal partner is a value. So now everyone can create better representations of themselves that help them do what they want to do.

Hopefully this helps explain how transformers work in a way that's a bit more easy to understand. I'm hopeful that having a model like this in your head will help a lot of the math to make more sense.

(A sidenote for those of you who are planning to dive into the math: depending on the text, "query", "key", and "value" can either refer to the original encodings, the learned weight matrices that multiply the encodings, or the value of the original encodings multiplied by the weights. Usually $Q$, $K$ and $V$ either refer to the original encodings (as I have done here) or the encodings multiplied by the weights (as [the transformer paper](https://arxiv.org/abs/1706.03762) does), and $W^Q$, $W^K$, $W^V$ refer to the learned weight matrices, but you often have to figure it out by context. There's also some tricks that people use like normalization and heads etc., but you can also find those details explained well [here](https://london-lowmanstone.medium.com/a-simpler-description-of-the-attention-mechanism-573e9e7f42b0).)

(And last, but certainly not least, my sincere apologies if it hurts (personally or systematically) to read yet another pretty classicly cishet example as an explainer; I unfortunately decided to prioritize this explanation's clarity over its inclusivity by including these gender and preference stereotypes to help readers keep track of the analogy. If you have a suggestion of how to improve the inclusivity without degrading the clarity, I would greatly appreciate it. Or if you just want to indicate that I should do better, a downvote will do that just fine. I'm not happy about the tradeoff either.)