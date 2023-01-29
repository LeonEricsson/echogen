import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Intuitive explanation
# The self-attention layer is a way to communicate information between different positions in the sequence. However,
# remeber that there is no notion of position inherently in the head, this comes purely from the positional encoding.
# As this is an decoder model, we want to make sure that the model can only attend to the past. As such, this is a masked
# self-attention layer.
class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # to avoid having to create the lower triangular matrix every time, similar to batchnorm's running mean.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C) -> (B, T, head_size)
        q = self.query(x) # (B, T, C) -> (B, T, head_size)
        wei = torch.bmm(q, k.transpose(-2,-1)) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # Note: when q and k are unit variance, we want to make sure wei is also unit variance. Otherwise, upon initialization and
        # especially when the head size is large, the softmax will saturate and the gradients will be very small.
        wei = wei / (C ** 0.5) # scale by the square root of the head size.
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask out the upper triangular part, deleted in a encoder 
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, C) -> (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
    
# Intuitive explanation
# The multi-head attention layer is a way to parallelize the self-attention layer. We can think of it
# as a way to divide the attention into separate channels and then concatenate them back together.
# This has been shown to be more effective than a single self-attention head. 
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.droupout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C) -> (B, T, num_heads * head_size)
        out = self.droupout(self.proj(out))
        return out

# Intuitive explanation
# Without a linear layer to follow the self-attention, we don't allow the model to use the information
# it learnt from the self-attention / communication step. We want to make sure to cultivate the information
# in a linear layer. 
class FeedForward(nn.Module):
    """ simple linear layer followed by non-linearity """
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
# Intuitive explanation
# Normalizing across the layer dimension as a further optimization improvement for in deep models.
class LayerNorm:
    def __init__(self, n_embd, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(n_embd)
        self.beta = torch.zeros(n_embd)
    def forward(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True, unbiased=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        return self.gamma * xhat + self.beta
    

# Intuitive explanation
# Combining the communication and computation into one block for simplicity. Note the residual
# connections in the forward pass. This helps optimization, especially when the bi-gram model
# becomes deeper. Creates a unimpeded gradient highway.
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd, n_head) -> None:
        # n_embd: number of dimensions in the input, n_head: number of heads in the self-attention
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        
    def forward(self, x):
        # Normalize before the attention and feed-forward layers as opposed to after (which was the original proposition).
        # This has turned out to be more popular in later days. The normalization occurs across the n_embd dimension meaning
        # both batch and time act as a batch dimension. Can be thought of as a per token normalization.
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x)) # residual connection
        return x

class GPT(nn.Module):
    """ character level language model using an autoregressive transformer encoder only architecture """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # n_layer blocks of n_head heads of 8-dimensional self-attention
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))