import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# Step 1: Sample Sentence
# -------------------------------

sentence = ["I", "love", "deep", "learning"]


# -------------------------------
# Step 2: Raw Attention Scores
# -------------------------------

attention_scores = torch.tensor([0.1, 0.3, 0.4, 0.2])


# -------------------------------
# Step 3: Apply Softmax
# -------------------------------

attention_weights = F.softmax(attention_scores, dim=0).detach().numpy()


# -------------------------------
# Step 4: Display Weights
# -------------------------------

print("Words:", sentence)
print("Attention Weights:", attention_weights)


# -------------------------------
# Step 5: Visualization (Heatmap)
# -------------------------------

plt.figure(figsize=(8, 2))

sns.heatmap(
    [attention_weights],
    annot=True,
    cmap="coolwarm",
    xticklabels=sentence,
    yticklabels=["Attention"]
)

plt.title("Attention Mechanism Visualization")
plt.show()
