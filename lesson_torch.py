"""Day 1: fit y=2x+1 and explain each line of the parameter update."""
import torch
from torch import nn

torch.manual_seed(42)
# The first learning exercise stays on CPU to keep device setup separate.
x = torch.tensor([[0.], [1.], [2.], [3.]])
y = 2 * x + 1
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

print("Input shape:", tuple(x.shape))
for step in range(300):
    prediction = model(x)
    loss = ((prediction - y) ** 2).mean()
    optimizer.zero_grad()  # Clear gradients accumulated by the previous step.
    loss.backward()       # Compute derivatives with the chain rule.
    optimizer.step()      # Update the parameters using their gradients.
    if step % 50 == 0:
        print(f"Step {step:3d} | loss={loss.item():.6f}")

print("Learned weight (target 2):", model.weight.item())
print("Learned bias (target 1):", model.bias.item())
print("Exercise: print weight.grad after backward(); explain zero_grad/backward/step.")
