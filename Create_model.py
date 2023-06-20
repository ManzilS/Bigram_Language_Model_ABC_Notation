import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# Initialize some hyperparameters
# Hyperparameters control various aspects of training, such as the batch size, 
# learning rate, dropout ratio, etc.
batch_size = 132 # Number of sequences to process in parallel.
block_size = 16 # Maximum context length for predictions.
max_iters = 500000 # Maximum number of iterations to train.
eval_interval = 100 # Frequency at which to evaluate the model.
learning_rate = 1e-3 # Learning rate for the optimizer.
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to use for computations, use GPU if available else CPU.
eval_iters = 100 # Number of evaluation iterations.
n_embd = 64 # Size of the embeddings.
n_head = 4 # Number of attention heads.
n_layer = 4 # Number of transformer layers.
dropout = 0.1 # Dropout ratio.

torch.manual_seed(1337) # Set the seed for reproducibility.

# Load the training text file.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract the unique characters in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars) # Size of the vocabulary.

# Map characters to indices and vice versa for easy conversion.
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Function to convert characters to indices.
decode = lambda l: ''.join([itos[i] for i in l]) # Function to convert indices to characters.

# Convert the whole dataset to indices and split it into training and validation sets.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# A function to get a batch of data with inputs x and targets y.
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Random indices for getting the batches.
    x = torch.stack([data[i:i+block_size] for i in ix]) # Inputs for the model.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Corresponding targets for the model.
    x, y = x.to(device), y.to(device) # Move tensors to the computation device.
    return x, y

# This function estimates the loss on both training and validation data without updating the model parameters.
def estimate_loss():
    out = {}
    model.eval() # Set the model to evaluation mode.
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set the model back to training mode.
    return out

# Transformer architecture components.

# Self-attention head.
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Key, query and value matrices for attention mechanism.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower triangle matrix for masking future tokens.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # Dropout layer.
        self.dropout = nn.Dropout(dropout)

    # Forward pass of the attention head.
    def forward(self, x):
        B, T, C = x.shape # Batch size, sequence length, embedding dimension.
        k = self.key(x) # Key.
        q = self.query(x) # Query.
        # Compute attention scores.
        wei = q @ k.transpose(-2,-1) * C**-0.5 # Scaling factor for the scores.
        # Mask the future tokens.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Compute attention weights using softmax.
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Compute the output by weighted aggregation of values.
        v = self.value(x)
        out = wei @ v
        return out

# Multiple heads of self-attention in parallel.
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # List of attention heads.
        self.proj = nn.Linear(n_embd, n_embd) # Final projection layer.
        self.dropout = nn.Dropout(dropout)

    # Forward pass of the multi-head attention layer.
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate the outputs of all attention heads.
        out = self.dropout(self.proj(out)) # Final projection and dropout.
        return out

# Feed-forward network.
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # A simple linear layer followed by a ReLU non-linearity and another linear layer.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    # Forward pass of the feed-forward network.
    def forward(self, x):
        return self.net(x)

# Transformer block: multi-head attention followed by feed-forward network.
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # Size of each head.
        self.sa = MultiHeadAttention(n_head, head_size) # Self-attention layer.
        self.ffwd = FeedFoward(n_embd) # Feed-forward network.
        self.ln1 = nn.LayerNorm(n_embd) # Layer normalization.
        self.ln2 = nn.LayerNorm(n_embd) # Layer normalization.

    # Forward pass of the transformer block.
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Add & Norm for self-attention.
        x = x + self.ffwd(self.ln2(x)) # Add & Norm for feed-forward network.
        return x

# The main model: a bigram language model.
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding layers for tokens and positions.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Multiple transformer blocks.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm.
        self.lm_head = nn.Linear(n_embd, vocab_size) # Linear layer for generating logits for the next token.

    # Forward pass of the language model.
    def forward(self, idx, targets=None):
        B, T = idx.shape # Batch size, sequence length.
        tok_emb = self.token_embedding_table(idx) # Token embeddings.
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Position embeddings.
        x = tok_emb + pos_emb # Add the token and position embeddings.
        x = self.blocks(x) # Pass through the transformer blocks.
        x = self.ln_f(x) # Final layer norm.
        logits = self.lm_head(x) # Compute the logits for the next token.

        # If targets are provided, compute the loss.
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # Compute cross-entropy loss.
        else:
            loss = None
        return logits, loss

    # Function for generating new text.
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # Current context.
            logits, loss = self(idx_cond) # Get predictions.
            logits = logits[:, -1, :] # Focus on the last time step.
            probs = F.softmax(logits, dim=-1) # Convert logits to probabilities.
            idx_next = torch.multinomial(probs, num_samples=1) # Sample the next token.
            idx = torch.cat((idx, idx_next), dim=1) # Update the current context.
        return idx

# Create the model and move it to the device.
model = BigramLanguageModel()
m = model.to(device)

# Print the number of parameters in the model.
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Create an optimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Main training loop.
for iter in range(max_iters):
    # Evaluate the loss on train and val sets periodically.
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data.
    xb, yb = get_batch('train')

    # Evaluate the loss.
    logits, loss = model(xb, yb)
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate new text from the model.
torch.save(model.state_dict(), './model/model.pt')
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
