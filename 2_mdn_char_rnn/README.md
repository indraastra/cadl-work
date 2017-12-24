## Goal

To train a "visual" char-rnn using a standard char-rnn model modified to use
visual embeddings as inputs and outputs and augmented with an MDN rather than
the usual softmax loss. 

### VAEGAN
* input: bitmap
* output: bitmap
* byproduct: an encoder, decoder, and visual embedding


### LSTM + MDN head
* input: previous embedding
* output: next embedding

### Functions
render(font, character) -> image
encode(image) -> embedding
decode(embedding) -> image
predict(embedding) -> embedding


### Procedure

First, a VAEGAN is trained on images produced by render(font, character) Ɐ
fonts and characters. This results in encode(image) and decode(image) functions
that work with visual embeddings of characters.

These embeddings are used as inputs and outputs to a char-rnn style model, but
with an MDN output and loss function. This allows us to manipulate the input
embeddings like we might play with vectors in latent space, while synthesizing
output that is nondeterministic in a way that the usual softmax-style output
for a char-rnn model is not. This allows both the style and substance to be
manipulated in the output vector, and the resulting embedding can be turned
into an image using decode().

### Future work

Use non-bitmap-based approach to synthesize (eg. LSTMs like otoro blog) at
arbitrary resolutions.
