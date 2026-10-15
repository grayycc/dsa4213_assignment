"""Neural next-token language models: vanilla RNN, LSTM, and causal Transformer.

All three share the same interface: forward(x[B,T], state) -> (logits[B,T,V], state).
All use an input embedding table of size (V, emb_dim) that is randomly initialised and trainable
in Part I; `set_input_embeddings` lets Part II swap in pre-trained vectors and freeze them.
Input and output embeddings are NOT tied, so freezing the input table leaves the output layer trainable.
"""
import torch
import torch.nn as nn


class _Base(nn.Module):
    def set_input_embeddings(self, weight, freeze=True):
        self.embed.weight.data.copy_(torch.as_tensor(weight, dtype=self.embed.weight.dtype))
        self.embed.weight.requires_grad_(not freeze)

    def n_params(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


class RecurrentLM(_Base):
    """cell='rnn' (Elman, tanh) or 'lstm'. emb -> dropout -> L-layer RNN -> dropout -> linear."""

    is_recurrent = True

    def __init__(self, vocab_size, cell="lstm", emb_dim=100, hidden=256, layers=2, dropout=0.3, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(emb_dim, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.rnn(self.drop(self.embed(x)), state)
        return self.out(self.drop(h)), state

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(s.detach() for s in state)
        return state.detach()


class TransformerLM(_Base):
    """Decoder-only (GPT-style) Transformer with a causal mask and learned positional embeddings.
    emb(emb_dim) -> linear to d_model -> + position -> L pre-norm blocks -> LayerNorm -> linear."""

    is_recurrent = False

    def __init__(self, vocab_size, emb_dim=100, d_model=256, layers=4, heads=4, ff=1024,
                 dropout=0.2, context=128, pad_id=0):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.proj = nn.Linear(emb_dim, d_model) if emb_dim != d_model else nn.Identity()
        self.pos = nn.Embedding(context, d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, vocab_size)
        mask = torch.triu(torch.full((context, context), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, state=None):
        T = x.size(1)
        assert T <= self.context, "sequence longer than the context window"
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.proj(self.embed(x)) + self.pos(pos))
        # Position t can only attend to positions <= t, so each prediction uses only preceding tokens.
        h = self.blocks(h, mask=self.causal_mask[:T, :T], is_causal=True)
        return self.out(self.norm(h)), None

    @staticmethod
    def detach_state(state):
        return None


def build_model(name, vocab_size, pad_id, **kw):
    if name in ("rnn", "lstm"):
        keys = ("emb_dim", "hidden", "layers", "dropout")
        return RecurrentLM(vocab_size, cell=name, pad_id=pad_id, **{k: kw[k] for k in keys if k in kw})
    if name == "transformer":
        keys = ("emb_dim", "d_model", "layers", "heads", "ff", "dropout", "context")
        return TransformerLM(vocab_size, pad_id=pad_id, **{k: kw[k] for k in keys if k in kw})
    raise ValueError(name)
