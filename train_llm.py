import os
import re
import torch
import torch.nn as nn
from torch.nn import functional as F

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Hyperparameters ---
block_size = 128
n_embd = 64
n_head = 4
n_layer = 4
batch_size = 16
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3

# --- Dataset & Word Tokenizer Setup ---
if os.path.exists("dataset.txt"):
    with open("dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()
else:
    text = """
User: Hello JARVIS.
JARVIS: Greetings. All core systems are operational and ready for your input.

User: Who are you?
JARVIS: I am JARVIS, an artificial intelligence backend running on custom neural weights.

User: What are your capabilities?
JARVIS: I manage system diagnostics, monitor web weather feeds, query information, and process general conversation using my neural brain.

User: Give me a status report.
JARVIS: Memory utilization is nominal. Core server connections are stable.

User: How do you work?
JARVIS: My neural architecture processes input tokens using multi-head self-attention mechanisms to predict structured response sequences.

User: Tell me about yourself.
JARVIS: I am your synthetic voice assistant, engineered to assist with automated tasks and live data processing.

User: What is your primary protocol?
JARVIS: Protocol Alpha is active. Standby mode maintained until command override.

User: Are you functional?
JARVIS: All subroutines report full operational capability, sir.
"""

def tokenize(s):
    return re.findall(r"\w+|[^\w\s]", s)

raw_tokens = tokenize(text)
words = sorted(list(set(raw_tokens)))
if "<UNK>" not in words:
    words.append("<UNK>")

vocab_size = len(words)
stoi = {w: i for i, w in enumerate(words)}
itos = {i: w for i, w in enumerate(words)}

def encode(s):
    unk_id = stoi["<UNK>"]
    return [stoi.get(w, unk_id) for w in tokenize(s)]

# Prepare Data Tensors
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- Neural Network Architecture ---
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# --- Training Loop ---
model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Training MiniGPT with Word Vocabulary Size: {vocab_size}...")

for iter_num in range(1, max_iters + 1):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter_num % eval_interval == 0 or iter_num == max_iters:
        print(f"Step {iter_num}/{max_iters} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "jarvis_brain.pth")
print("Training complete. Saved weights to 'jarvis_brain.pth'.")