# ===== IMPORTS =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ===== REPRODUCIBILITY & DEVICE =====
# Fix random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

# ===== HYPERPARAMETERS =====
BATCH_SIZE = 128           # Number of samples per batch
EPOCHS = 100               # Total training epochs
LEARNING_RATE = 5e-4       # Learning rate for optimizer
PATCH_SIZE = 4             # Patch size for ViT
NUM_CLASSES = 10           # CIFAR-10 has 10 classes
IMAGE_SIZE = 32            # CIFAR-10 image size
CHANNELS = 3               # RGB images
EMBED_DIM = 384            # Embedding dimension
NUM_HEADS = 8              # Number of attention heads
DEPTH = 12                 # Number of transformer layers
MLP_RATIO = 4              # Expansion ratio in MLP
DROP_RATE = 0.1            # Dropout probability
ATTN_DROP_RATE = 0.1       # Attention dropout probability
WEIGHT_DECAY = 0.05        # Regularization for optimizer
WARMUP_EPOCHS = 5          # Warm-up period for LR scheduler

# ===== DATA AUGMENTATION & NORMALIZATION =====
# Augmentations for training data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Normalization only for test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# ===== LOAD CIFAR-10 DATASETS =====
train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ===== CUSTOM COLOR MAP FOR VISUALIZATION =====
def create_custom_cmap():
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
    return LinearSegmentedColormap.from_list("custom", colors)

custom_cmap = create_custom_cmap()

# ===== PATCH EMBEDDING LAYER =====
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Learnable CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # Learnable position embedding

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # Shape: (B, Num_Patches, Embed_Dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Repeat CLS token for each item in batch
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate CLS token
        x = x + self.pos_embed  # Add position embedding
        return x

# ===== MLP BLOCK =====
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ===== TRANSFORMER ENCODER BLOCK =====
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.attention_weights = None  # For visualization

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm)
        self.attention_weights = attn_weights  # Save attention for visualization
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ===== VISION TRANSFORMER MODEL =====
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=384, depth=12, num_heads=8, mlp_ratio=4,
                 drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_rate, depth)]  # Stochastic depth rates
        self.blocks = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, True,
                                    drop_rate, attn_drop_rate, dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  # Final classifier head

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # Return only CLS token output

# ===== MODEL, LOSS, OPTIMIZER, SCHEDULER =====
model = VisionTransformer(...).to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    [
        optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS),
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    ],
    milestones=[WARMUP_EPOCHS]
)


# ===== TRAINING FUNCTION =====
def train(model, loader, optimizer, criterion):
    model.train()  # Set model to training mode
    total_loss, correct = 0, 0  # Initialize accumulators

    for x, y in loader:
        x, y = x.to(device), y.to(device)  # Move data to device

        optimizer.zero_grad()  # Reset gradients
        out = model(x)  # Forward pass
        loss = criterion(out, y)  # Compute loss

        loss.backward()  # Backpropagate
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to avoid explosion
        optimizer.step()  # Update weights

        total_loss += loss.item() * x.size(0)  # Accumulate weighted loss
        correct += (out.argmax(1) == y).sum().item()  # Count correct predictions

    scheduler.step()  # Update learning rate schedule
    return total_loss / len(loader.dataset), correct / len(loader.dataset)  # Average loss and accuracy


# ===== EVALUATION FUNCTION =====
def evaluate(model, loader, return_predictions=False):
    model.eval()  # Evaluation mode
    correct = 0
    all_preds = []
    all_targets = []

    with torch.inference_mode():  # Disable gradient computation
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()

            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

    if return_predictions:
        return correct / len(loader.dataset), (all_preds, all_targets)
    return correct / len(loader.dataset)


# ===== TRAINING CURVE PLOT =====
def plot_training_curves(train_losses, train_accs, test_accs, lr_history):
    plt.figure(figsize=(18, 6))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, color='#2E86AB', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, color='#A23B72', linewidth=2, label='Train')
    plt.plot(test_accs, color='#F18F01', linewidth=2, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Learning rate schedule
    plt.subplot(1, 3, 3)
    plt.plot(lr_history, color='#3B1F2B', linewidth=2)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ===== CONFUSION MATRIX PLOT =====
def plot_confusion_matrix(model, loader, class_names):
    model.eval()
    _, (all_preds, all_targets) = evaluate(model, loader, return_predictions=True)

    cm = confusion_matrix(all_targets, all_preds)  # Compute confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


# ===== PREDICTION VISUALIZATION =====
def visualize_predictions(model, dataset, classes, n_images=9):
    model.eval()
    plt.figure(figsize=(12, 12))
    indices = random.sample(range(len(dataset)), n_images)

    for i, idx in enumerate(indices):
        img, true_label = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        # Unnormalize image
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img, 0, 1)

        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        truth = classes[true_label] == classes[predicted.item()]
        color = "#2E86AB" if truth else "#C73E1D"
        title = f"True: {classes[true_label]}\nPred: {classes[predicted.item()]}\nConf: {confidence.item():.2f}"
        plt.title(title, color=color, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ===== ATTENTION MAP VISUALIZATION =====
def visualize_attention(model, dataset, idx=0, layer=0):
    model.eval()
    img, label = dataset[idx]
    input_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        x = model.patch_embed(input_tensor)
        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if i == layer:
                attn_weights = blk.attention_weights

        # Process attention weights
        attn_weights = attn_weights.mean(dim=1)  # Average over heads
        cls_attn = attn_weights[0, 0, 1:]  # Get attention from CLS token to patches
        grid_size = int(np.sqrt(cls_attn.size(0)))
        attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()

    # Plot original image and attention
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Unnormalize image
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465])
    img = np.clip(img, 0, 1)

    ax1.imshow(img)
    ax1.set_title(f"Original Image\nLabel: {dataset.classes[label]}", fontsize=12)
    ax1.axis('off')

    im = ax2.imshow(attn_map, cmap=custom_cmap, interpolation='lanczos')
    ax2.set_title(f"Attention Map (Layer {layer})", fontsize=12)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    plt.tight_layout()
    plt.show()


# ===== MAIN TRAINING LOOP =====
def train_model():
    train_losses = []
    train_accs = []
    test_accs = []
    lr_history = []

    best_acc = 0.0
    best_model = None

    for epoch in tqdm(range(EPOCHS), desc="Training"):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model.state_dict().copy()

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc*100:.2f}%, "
                  f"Test Acc: {test_acc*100:.2f}%")

    model.load_state_dict(best_model)  # Load best weights
    plot_training_curves(train_losses, train_accs, test_accs, lr_history)
    return train_losses, train_accs, test_accs


# ===== RUN TRAINING AND VISUALIZATION =====
train_losses, train_accs, test_accs = train_model()  # Train the model

plot_confusion_matrix(model, test_loader, train_dataset.classes)  # Show confusion matrix

visualize_predictions(model, test_dataset, train_dataset.classes)  # Show sample predictions


