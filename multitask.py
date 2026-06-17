import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

SEED = 21
torch.manual_seed(SEED)
np.random.seed(SEED)

char_to_idx = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
    'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,
    't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, ' ': 26, '!': 27, '#': 28
}
idx_to_char = {v: k for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx)

START_TOKEN = 27
END_TOKEN = 28
PAD_TOKEN = 26

class MultiTaskLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 64, 
                 embedding_dim: int = 32, num_layers: int = 1, 
                 dropout_rate: float = 0.3):
        super(MultiTaskLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_TOKEN)
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc_letter = nn.Linear(hidden_dim, vocab_size)
        
        self.fc_gender = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None):
        embedded = self.dropout(self.embedding(x))
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        letter_logits = self.fc_letter(lstm_out[:, :-1, :])
        
        batch_size = x.size(0)
        gender_features = []
        
        for i in range(batch_size):
            end_positions = (x[i] == END_TOKEN).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                last_pos = end_positions[0]  
                gender_features.append(lstm_out[i, last_pos, :])
            else:
                gender_features.append(lstm_out[i, -1, :]) 
        
        gender_features = torch.stack(gender_features, dim=0)
        gender_logits = self.fc_gender(gender_features)
        
        return letter_logits, gender_logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        )
    
class MultiTaskTrainer:
    
    def __init__(self, model: nn.Module, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device)
        self.device = device
        
        self.train_loss = {'letter': [], 'gender': [], 'total': []}
        self.val_loss = {'letter': [], 'gender': [], 'total': []}
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _compute_losses(self, letter_logits, targets, gender_logits, gender_targets, 
                       criterion_letter, criterion_gender, gender_weight):
        batch_size, seq_len, vocab_size = letter_logits.shape
        
        mask = (targets != PAD_TOKEN).float()
        
        letter_loss_per_sample = criterion_letter(
            letter_logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        ).reshape(batch_size, seq_len)
        
        letter_loss = (letter_loss_per_sample * mask).sum() / (mask.sum() + 1e-8)
        
        gender_loss = criterion_gender(gender_logits.squeeze(), gender_targets)
        
        return letter_loss, gender_loss
    
    def train_epoch(self, dataloader, optimizer, criterion_letter, 
                   criterion_gender, gender_weight):
        self.model.train()
        
        total_letter_loss = 0.0
        total_gender_loss = 0.0
        num_batches = 0
        
        for x, targets, gender_targets in dataloader:
            x = x.to(self.device)
            targets = targets.to(self.device)
            gender_targets = gender_targets.to(self.device)
            
            hidden = self.model.init_hidden(x.size(0), self.device)
            
            letter_logits, gender_logits, _ = self.model(x, hidden)
            
            letter_loss, gender_loss = self._compute_losses(
                letter_logits, targets, gender_logits, gender_targets,
                criterion_letter, criterion_gender, gender_weight
            )
            
            total_loss = letter_loss + gender_weight * gender_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_letter_loss += letter_loss.item()
            total_gender_loss += gender_loss.item()
            num_batches += 1
        
        return total_letter_loss / num_batches, total_gender_loss / num_batches
    
    def validate(self, dataloader, criterion_letter, criterion_gender, gender_weight):
        self.model.eval()
        
        total_letter_loss = 0.0
        total_gender_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, targets, gender_targets in dataloader:
                x = x.to(self.device)
                targets = targets.to(self.device)
                gender_targets = gender_targets.to(self.device)
                
                hidden = self.model.init_hidden(x.size(0), self.device)
                letter_logits, gender_logits, _ = self.model(x, hidden)
                
                letter_loss, gender_loss = self._compute_losses(
                    letter_logits, targets, gender_logits, gender_targets,
                    criterion_letter, criterion_gender, gender_weight
                )
                
                total_letter_loss += letter_loss.item()
                total_gender_loss += gender_loss.item()
                num_batches += 1
        
        return total_letter_loss / num_batches, total_gender_loss / num_batches
    
    def fit(self, train_loader, val_loader, epochs=100, lr=0.001, 
            gender_weight=0.3, early_stopping_patience=10):
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion_letter = nn.CrossEntropyLoss(reduction='none', ignore_index=PAD_TOKEN)
        criterion_gender = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            train_letter_loss, train_gender_loss = self.train_epoch(
                train_loader, optimizer, criterion_letter, criterion_gender, gender_weight
            )
            
            val_letter_loss, val_gender_loss = self.validate(
                val_loader, criterion_letter, criterion_gender, gender_weight
            )
            
            train_total = train_letter_loss + gender_weight * train_gender_loss
            val_total = val_letter_loss + gender_weight * val_gender_loss
            
            self.train_loss['letter'].append(train_letter_loss)
            self.train_loss['gender'].append(train_gender_loss)
            self.train_loss['total'].append(train_total)
            self.val_loss['letter'].append(val_letter_loss)
            self.val_loss['gender'].append(val_gender_loss)
            self.val_loss['total'].append(val_total)
            
            scheduler.step(val_total)
            
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(f"\nРанняя остановка на эпохе {epoch}")
                    break
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Эпоха {epoch:3d} | "
                      f"Train L: {train_letter_loss:.4f}, G: {train_gender_loss:.4f} | "
                      f"Val L: {val_letter_loss:.4f}, G: {val_gender_loss:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    def plot_convergence(self):
        epochs = range(1, len(self.train_loss['letter']) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        titles = ['Letter Prediction Loss', 'Gender Prediction Loss', 'Total Loss']
        train_metrics = [self.train_loss['letter'], self.train_loss['gender'], self.train_loss['total']]
        val_metrics = [self.val_loss['letter'], self.val_loss['gender'], self.val_loss['total']]
        
        for ax, title, train_metric, val_metric in zip(axes, titles, train_metrics, val_metrics):
            ax.plot(epochs, train_metric, 'b-', label='Train', linewidth=2)
            ax.plot(epochs, val_metric, 'r-', label='Validation', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_name(self, temperature=0.8, max_length=20):
        self.model.eval()
        
        with torch.no_grad():
            name_indices = [START_TOKEN]
            hidden = self.model.init_hidden(1, self.device)
            
            for _ in range(max_length):
                x = torch.tensor([[name_indices[-1]]]).to(self.device)
                
                embedded = self.model.embedding(x)
                lstm_out, hidden = self.model.lstm(embedded, hidden)
                letter_logits = self.model.fc_letter(lstm_out[:, -1, :])
                
                probs = torch.softmax(letter_logits / temperature, dim=-1)
                
                probs[0, PAD_TOKEN] = 0
                probs = probs / probs.sum()
                
                next_idx = torch.multinomial(probs, 1).item()
                name_indices.append(next_idx)
                
                if next_idx == END_TOKEN:
                    break
            
            name = ''.join([
                idx_to_char.get(idx, '?') 
                for idx in name_indices[1:] 
                if idx not in [END_TOKEN, PAD_TOKEN]
            ])
            
        return name
    
    def predict_gender(self, name_encoded):
        self.model.eval()
        
        seq = torch.tensor([name_encoded], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            hidden = self.model.init_hidden(1, self.device)
            _, gender_logits, _ = self.model(seq, hidden)
            gender_prob = torch.sigmoid(gender_logits).item()
        
        return gender_prob
    
    def get_embeddings(self):
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for idx in range(vocab_size):
                emb = self.model.embedding(torch.tensor([idx]).to(self.device))
                embeddings.append(emb.cpu().numpy().flatten())
        
        return np.array(embeddings)
