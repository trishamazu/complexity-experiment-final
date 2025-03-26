import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class ComplexityDataset(Dataset):
    """Dataset for complexity prediction from embeddings"""
    def __init__(self, csv_path, image_dir=None, transform=None):
        """
        Args:
            csv_path: Path to the CSV file containing image names, complexity scores, and embeddings
            image_dir: Directory to the images (not used for embedding-based model but kept for compatibility)
            transform: Optional transform to be applied to images (not used for embedding-based model)
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        
        # Extract image names, targets, and embeddings
        self.image_names = self.data.iloc[:, 0].values
        self.targets = self.data.iloc[:, 1].values.astype(np.float32)
        self.embeddings = self.data.iloc[:, 2:].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        image_name = self.image_names[idx]
        
        sample = {
            'embedding': embedding,
            'target': target,
            'image_name': image_name
        }
        
        return sample

class CrossDimensionalAttention(nn.Module):
    """Cross-dimensional attention module to capture relationships between embedding dimensions"""
    def __init__(self, embed_dim, num_heads=4):
        super(CrossDimensionalAttention, self).__init__()
        # Adjust embed_dim to make it divisible by num_heads if needed
        self.original_embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            # Pad the embedding dimension to make it divisible by num_heads
            self.embed_dim = ((embed_dim // num_heads) + 1) * num_heads
            self.projection = nn.Linear(embed_dim, self.embed_dim)
            self.projection_out = nn.Linear(self.embed_dim, embed_dim)
            print(f"Adjusted embedding dimension from {embed_dim} to {self.embed_dim} to be divisible by {num_heads} heads")
        else:
            self.embed_dim = embed_dim
            self.projection = None
            self.projection_out = None
        
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Project to adjusted dimension if needed
        if self.projection is not None:
            x = self.projection(x)
        
        # Reshape x to [batch_size, 1, embed_dim] to represent each embedding as a sequence of length 1
        x = x.unsqueeze(1)  
        
        # Apply layer normalization
        residual = x
        x = self.layer_norm(x)
        
        # Project queries, keys, and values
        q = self.query(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        k = self.key(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)    # [batch_size, num_heads, 1, head_dim]
        v = self.value(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, 1, 1]
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, 1, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.embed_dim)  # [batch_size, 1, embed_dim]
        
        # Output projection
        output = self.out_proj(context)
        
        # Residual connection
        output = output + residual
        
        # Remove sequence dimension and return [batch_size, embed_dim]
        output = output.squeeze(1)
        
        # Project back to original dimension if needed
        if self.projection_out is not None:
            output = self.projection_out(output)
            
        return output

class ComplexityPredictor(nn.Module):
    """Neural network for predicting complexity scores from embeddings"""
    def __init__(self, embed_dim, hidden_dim=128, num_heads=4):
        super(ComplexityPredictor, self).__init__()
        
        self.attention = CrossDimensionalAttention(embed_dim, num_heads)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # Apply cross-dimensional attention
        x = self.attention(x)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        
        return x.squeeze()  # Output shape: [batch_size]

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience=5, checkpoint_path='best_model.pt'):
    """Train the model with early stopping"""
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as t:
            for batch in t:
                embeddings = batch['embedding'].to(device)
                targets = batch['target'].to(device)
                
                optimizer.zero_grad()
                outputs = model(embeddings)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * embeddings.size(0)
                t.set_postfix(loss=loss.item())
                
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]") as t:
                for batch in t:
                    embeddings = batch['embedding'].to(device)
                    targets = batch['target'].to(device)
                    
                    outputs = model(embeddings)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * embeddings.size(0)
                    t.set_postfix(loss=loss.item())
                    
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on the test set"""
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as t:
            for batch in t:
                embeddings = batch['embedding'].to(device)
                batch_targets = batch['target'].to(device)
                
                outputs = model(embeddings)
                loss = criterion(outputs, batch_targets)
                
                test_loss += loss.item() * embeddings.size(0)
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                
    test_loss /= len(test_loader.dataset)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    print(f'Test Loss: {test_loss:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
    
    return {
        'test_loss': test_loss,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }

def plot_results(history, results):
    """Plot training curves and prediction results"""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot predictions vs targets
    ax2.scatter(results['targets'], results['predictions'])
    ax2.plot([min(results['targets']), max(results['targets'])], 
             [min(results['targets']), max(results['targets'])], 'r--')
    ax2.set_xlabel('True Complexity Score')
    ax2.set_ylabel('Predicted Complexity Score')
    ax2.set_title(f'Predictions vs Targets (R² = {results["r2"]:.4f})')
    
    plt.tight_layout()
    plt.savefig('complexity_prediction_results.png')
    plt.close()

def main(config):
    """Main function to run the training pipeline"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data = pd.read_csv(config['csv_path'])
    
    # Determine the embedding dimension from the CSV
    embed_dim = data.shape[1] - 2  # Subtract 2 for image name and target columns
    print(f"Detected embedding dimension: {embed_dim}")
    
    # Split into train, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Save splits to temporary CSV files
    train_csv_path = 'train_temp.csv'
    val_csv_path = 'val_temp.csv'
    test_csv_path = 'test_temp.csv'
    
    train_data.to_csv(train_csv_path, index=False)
    val_data.to_csv(val_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)
    
    # Create datasets
    train_dataset = ComplexityDataset(train_csv_path, config['image_dir'])
    val_dataset = ComplexityDataset(val_csv_path, config['image_dir'])
    test_dataset = ComplexityDataset(test_csv_path, config['image_dir'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Initialize model
    model = ComplexityPredictor(
        embed_dim=embed_dim,
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4)
    ).to(device)
    
    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    
    # Train model
    checkpoint_path = config.get('checkpoint_path', 'best_complexity_model.pt')
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs=config['epochs'],
        patience=config.get('patience', 5),
        checkpoint_path=checkpoint_path
    )
    
    # Evaluate model
    results = evaluate_model(model, test_loader, criterion, device)
    
    # Plot and save results
    plot_results(history, results)
    
    # Clean up temporary files
    for path in [train_csv_path, val_csv_path, test_csv_path]:
        if os.path.exists(path):
            os.remove(path)
    
    print("Training completed!")
    
    return model, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train complexity predictor model')
    parser.add_argument('--config', type=str, default='./master_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['attention_training']
        
    main(config)