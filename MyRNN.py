import numpy as np
import matplotlib.pyplot as plt

class myRNN:
    def __init__(self, vocab_size, hidden_dim=64, embedding_dim=32, 
                 learning_rate=0.01, random_state=21, gender_loss='bce'):
        np.random.seed(random_state)
        self.lr = learning_rate
        self.hidden_dim = hidden_dim
        self.gender_loss = gender_loss
        self.vocab_size = vocab_size
        
        # Инициализация весов
        self.W_xh = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_hy = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.W_hg = np.random.randn(1, hidden_dim) * 0.01
        
        self.b_h = np.zeros(hidden_dim)
        self.b_g = np.zeros(1)
        
        # Embedding слой
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Кеш для backward pass
        self.cache = {}
        self.h = np.zeros(hidden_dim)
        
        # История обучения
        self.history = {'letter_train': [], 'gender_train': [], 
                       'letter_val': [], 'gender_val': []}
        
    def reset_state(self):
        self.h = np.zeros(self.hidden_dim)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))
    
    def softmax(self, logits): # преобразует вектор «сырых» оценок (логитов) в распределение вероятностей по классам
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / (np.sum(exp_logits) + 1e-9)
    
    def forward_rnn(self, x_idx, training=True):
        x = self.embedding[x_idx]
        
        z = self.W_xh @ x + self.W_hh @ self.h + self.b_h
        h_new = np.tanh(z) # новое скрытое состояние
        
        logits = self.W_hy @ h_new
        
        if training:
            self.cache = {
                'x': x,
                'h_prev': self.h.copy(),
                'z': z,
                'h_new': h_new
            }
        
        self.h = h_new
        return logits
    
    def backward_rnn(self, dL_dy):
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        z = self.cache['z']
        
        dL_dW_hy = np.outer(dL_dy, self.h)
        dL_dh = self.W_hy.T @ dL_dy
        
        dtanh = 1 - np.tanh(z)**2
        dL_dz = dL_dh * dtanh
        
        dL_dW_xh = np.outer(dL_dz, x)
        dL_dW_hh = np.outer(dL_dz, h_prev)
        dL_db_h = dL_dz
        
        self.W_hy -= self.lr * dL_dW_hy
        self.W_xh -= self.lr * dL_dW_xh
        self.W_hh -= self.lr * dL_dW_hh
        self.b_h -= self.lr * dL_db_h
        
        dL_demb = self.W_xh.T @ dL_dz
        return dL_demb
    
    def forward_gender(self):
        logit = self.W_hg @ self.h + self.b_g
        return logit[0]
    
    def compute_gender_loss(self, logit, label):
        if self.gender_loss == 'bce':
            # Binary Cross-Entropy с sigmoid
            prob = self.sigmoid(logit)
            loss = -(label * np.log(prob + 1e-9) + 
                    (1 - label) * np.log(1 - prob + 1e-9))
            grad = prob - label  # dL/dlogit
        
        else:  # 'nll'
            # Negative Log-Likelihood с softmax для 2 классов
            logits = np.array([0.0, logit])
            
            # log_softmax для численной стабильности
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            sum_exp = np.sum(exp_logits)
            log_probs = (logits - max_logit) - np.log(sum_exp)
            probs = exp_logits / sum_exp
            
            loss = -log_probs[int(label)]
            
            grad = probs[1] - (1.0 if label == 1 else 0.0)
        
        return loss, grad
    
    def backward_gender(self, grad):
        dL_dW_hg = grad * self.h.reshape(1, -1)
        dL_db_g = grad
        
        self.W_hg -= self.lr * dL_dW_hg
        self.b_g -= self.lr * dL_db_g
    
    def train_step(self, word, gender_label):
        self.reset_state()
        total_letter_loss = 0
        
        for t in range(len(word) - 1):
            x_t = word[t]
            y_next = word[t + 1]
            
            logits = self.forward_rnn(x_t, training=True)
            probs = self.softmax(logits)
            
            loss = -np.log(probs[y_next] + 1e-9)
            total_letter_loss += loss
            
            dL_dy = probs.copy()
            dL_dy[y_next] -= 1 # градиент потерь по логитам (входу softmax), т.е. ∂L/∂logits
            
            dL_demb = self.backward_rnn(dL_dy)
            self.embedding[x_t] -= self.lr * dL_demb
        
        gender_logit = self.forward_gender()
        
        gender_loss, grad = self.compute_gender_loss(gender_logit, gender_label)
        
        self.backward_gender(grad)
        
        avg_letter_loss = total_letter_loss / (len(word) - 1) if len(word) > 1 else 0
        return avg_letter_loss, gender_loss
    
    def evaluate(self, X, y):
        total_letter_loss = 0
        total_gender_loss = 0
        
        saved_h = self.h.copy()
        saved_cache = self.cache.copy()
        
        for word, gender_label in zip(X, y):
            self.reset_state()
            word_loss = 0
            
            for t in range(len(word) - 1):
                logits = self.forward_rnn(word[t], training=False)
                probs = self.softmax(logits)
                word_loss += -np.log(probs[word[t + 1]] + 1e-9)
            
            gender_logit = self.forward_gender()
            gender_loss, _ = self.compute_gender_loss(gender_logit, gender_label)
            
            total_letter_loss += word_loss / (len(word) - 1) if len(word) > 1 else 0
            total_gender_loss += gender_loss
        
        self.h = saved_h
        self.cache = saved_cache
        
        return total_letter_loss / len(X), total_gender_loss / len(X)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, verbose=True):
        for epoch in range(epochs):
            train_letter_loss = 0
            train_gender_loss = 0
            
            for word, gender in zip(X_train, y_train):
                l_loss, g_loss = self.train_step(word, gender)
                train_letter_loss += l_loss
                train_gender_loss += g_loss
            
            train_letter_loss /= len(X_train)
            train_gender_loss /= len(X_train)
            
            val_letter_loss, val_gender_loss = self.evaluate(X_val, y_val)
            
            self.history['letter_train'].append(train_letter_loss)
            self.history['gender_train'].append(train_gender_loss)
            self.history['letter_val'].append(val_letter_loss)
            self.history['gender_val'].append(val_gender_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train L: {train_letter_loss:.4f} G: {train_gender_loss:.4f} | "
                      f"Val L: {val_letter_loss:.4f} G: {val_gender_loss:.4f}")
    
    def generate_name(self, start_idx=27, temperature=1.0, max_length=20):
        self.reset_state()
        sequence = [start_idx]
        
        for _ in range(max_length):
            logits = self.forward_rnn(sequence[-1], training=False)
            probs = self.softmax(logits / temperature)
            next_idx = np.random.choice(self.vocab_size, p=probs) #случайно выбираем индекс с вероятностью, равной probs
            sequence.append(next_idx) 
            
            if next_idx == 28:
                break
        
        return sequence
    
    def predict_gender_for_name(self, word):
        self.reset_state()
        
        for idx in word:
            self.forward_rnn(idx, training=False)
        
        logit = self.forward_gender()
        
        if self.gender_loss == 'bce':
            prob = self.sigmoid(logit)
        else:  # 'nll'
            logits = np.array([0.0, logit])
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            prob = probs[1]  
        
        return prob
    
    def plot_convergence(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['letter_train']) + 1)
        
        axes[0].plot(epochs, self.history['letter_train'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['letter_val'], 'r-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Letter Prediction Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(epochs, self.history['gender_train'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['gender_val'], 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'Gender Prediction Loss ({self.gender_loss.upper()})')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()