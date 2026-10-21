# LAB 05 - Calculations and theory

## One RNN step

Using column vectors and the matrix layout in the handout:

\[
W_{xh}x_t =
\begin{bmatrix}0.5&0.2\\0.1&0.4\end{bmatrix}
\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}0.5\\0.1\end{bmatrix}.
\]

\[
W_{hh}h_{t-1} =
\begin{bmatrix}0.3&0.1\\0.2&0.5\end{bmatrix}
\begin{bmatrix}0.5\\0.2\end{bmatrix}
=\begin{bmatrix}0.3(0.5)+0.1(0.2)\\0.2(0.5)+0.5(0.2)\end{bmatrix}
=\begin{bmatrix}0.17\\0.20\end{bmatrix}.
\]

Thus, with zero bias, \(a_t=[0.67, 0.30]^T\) and
\(h_t=\tanh(a_t)\approx[0.58498, 0.29131]^T\).

## Information flow

For "I did not like this movie", the state after "I" can encode a speaker; after "did" it can encode an action frame; after "not" it can encode negation; after "like" it can combine that negation with the positive verb. "this" and "movie" can then update the target of that opinion. These are possible learned representations, not guaranteed human-readable slots. In \(h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)\), "not" affects the state at "like" through \(W_{hh}h_{t-1}\). Average Word2Vec uses one commutative mean, so it cannot encode this order by itself.

## Word order

"dog bites man" and "man bites dog" contain the same multiset of words, so their Average Word2Vec vectors are exactly equal when the same word vectors are used. A recurrent model computes its state after each token from its preceding state. The two sequences enter the same cell in different orders and can therefore yield different final states. Its capacity to distinguish them does not guarantee that a fitted model learns the correct roles.

## Why distant dependencies are difficult

Backpropagation through time multiplies Jacobians across time steps. Repeated factors with norm below one can make a distant token's gradient tiny; factors above one can make it large. Longer paths increase both risks. Gradient clipping limits explosion, while LSTM's gated cell state and GRU's update path make preservation of useful information easier. Neither architecture guarantees correct long-context reasoning.

## One LSTM cell update

\(c_t=f_tc_{t-1}+i_t\widetilde c_t=0.9(0.8)+0.2(0.5)=0.72+0.10=0.82\).
A forget gate near one retains most previous cell state; a gate near zero discards most of it. The input gate controls how much new candidate information is added.

## Architecture comparison

| Property | RNN | LSTM | GRU |
|---|---|---|---|
| Recurrent hidden state | Yes | Yes | Yes |
| Separate cell state | No | Yes | No |
| Gates | No | Forget, input, output | Reset, update |
| Distant dependencies | Often difficult | Better route for preservation | Better route for preservation |

All three process the input in order. Attention offers a more direct path between distant positions than a chain of recurrent state updates.
