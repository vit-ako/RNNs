import numpy as np
import matplotlib.pyplot as plt

class GRUCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # reset gate
        self.W_r = np.random.randn(hidden_dim, input_dim) * 0.01
        self.U_r = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_r = np.zeros(hidden_dim)
        
        # update gate
        self.W_z = np.random.randn(hidden_dim, input_dim) * 0.01
        self.U_z = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_z = np.zeros(hidden_dim)
        
        # candidat hidden state
        self.W_h = np.random.randn(hidden_dim, input_dim) * 0.01
        self.U_h = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros(hidden_dim)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x, h_prev):
        r = self.sigmoid(self.W_r @ x + self.U_r @ h_prev + self.b_r)
        z = self.sigmoid(self.W_z @ x + self.U_z @ h_prev + self.b_z)
        h_tilde = np.tanh(self.W_h @ x + self.U_h @ (r * h_prev) + self.b_h)
        h_new = (1 - z) * h_prev + z * h_tilde

        cache = {
            'x': x,
            'h_prev': h_prev,
            'r': r,
            'z': z,
            'h_tilde': h_tilde,
            'h_new': h_new
        }
        
        return h_new, cache
    
    def backward(self, dL_dh, cache, learning_rate):
        x = cache['x']
        h_prev = cache['h_prev']
        r = cache['r']
        z = cache['z']
        h_tilde = cache['h_tilde']
        
        dL_dz = dL_dh * (h_tilde - h_prev)
        dL_dh_tilde = dL_dh * z
        dL_dh_prev = dL_dh * (1 - z)
        
        dtanh = 1 - np.tanh(self.W_h @ x + self.U_h @ (r * h_prev) + self.b_h)**2
        dL_dh_tilde_raw = dL_dh_tilde * dtanh
        
        dL_dW_h = np.outer(dL_dh_tilde_raw, x)
        dL_dU_h = np.outer(dL_dh_tilde_raw, r * h_prev)
        dL_db_h = dL_dh_tilde_raw
        dL_dr = (self.U_h.T @ dL_dh_tilde_raw) * h_prev
        
        d_sigmoid_r = r * (1 - r)
        dL_dr_raw = dL_dr * d_sigmoid_r
        dL_dW_r = np.outer(dL_dr_raw, x)
        dL_dU_r = np.outer(dL_dr_raw, h_prev)
        dL_db_r = dL_dr_raw
        dL_dh_prev += self.U_r.T @ dL_dr_raw
        
        d_sigmoid_z = z * (1 - z)
        dL_dz_raw = dL_dz * d_sigmoid_z
        dL_dW_z = np.outer(dL_dz_raw, x)
        dL_dU_z = np.outer(dL_dz_raw, h_prev)
        dL_db_z = dL_dz_raw
        dL_dh_prev += self.U_z.T @ dL_dz_raw
        
        self.W_r -= learning_rate * dL_dW_r
        self.U_r -= learning_rate * dL_dU_r
        self.b_r -= learning_rate * dL_db_r
        
        self.W_z -= learning_rate * dL_dW_z
        self.U_z -= learning_rate * dL_dU_z
        self.b_z -= learning_rate * dL_db_z
        
        self.W_h -= learning_rate * dL_dW_h
        self.U_h -= learning_rate * dL_dU_h
        self.b_h -= learning_rate * dL_db_h
        
        dL_dx = (self.W_r.T @ dL_dr_raw + 
                 self.W_z.T @ dL_dz_raw + 
                 self.W_h.T @ dL_dh_tilde_raw)
        
        return dL_dx, dL_dh_prev


class GRURNN:
    def __init__(self, vocab_size, hidden_dim=128, embedding_dim=64, 
                 learning_rate=0.01, dropout_rate=0.0, random_state=21,
                 gender_loss_type='bce'):
        
        np.random.seed(random_state)
        self.random_state = random_state
        self.lr = learning_rate
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.gender_loss_type = gender_loss_type
        
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.gru_cell = GRUCell(embedding_dim, hidden_dim)
        
        self.W_y = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_y = np.zeros(vocab_size)
        
        self.W_gender = np.random.randn(1, hidden_dim) * 0.01
        self.b_gender = np.zeros(1)
        
        self.h = np.zeros(hidden_dim)
        self.caches = []
        self.char_indices = []
        
        self.train_loss_history = {'letter': [], 'gender': []}
        self.val_loss_history = {'letter': [], 'gender': []}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def dropout(self, x, training=True):
        if not training or self.dropout_rate == 0:
            return x
        mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
        return x * mask
    
    def reset_state(self):
        self.h = np.zeros(self.hidden_dim)
        self.caches = []
        self.char_indices = []
    
    def forward_gender(self):
        logit = (self.W_gender @ self.h + self.b_gender)[0]
        return logit
    
    def compute_gender_loss(self, logit, label):
        if self.gender_loss_type == 'bce':
            # Binary Cross-Entropy
            prob = self.sigmoid(logit)
            loss = -(label * np.log(prob + 1e-9) + 
                    (1 - label) * np.log(1 - prob + 1e-9))
            grad = prob - label
            return loss, grad
        
        else:  # 'nll'
            logits = np.array([0.0, logit])
            
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            sum_exp = np.sum(exp_logits)
            probs = exp_logits / sum_exp
            
            # NLL loss
            loss = -np.log(probs[int(label)] + 1e-9)
            
            # Градиент: dL/dlogit
            grad = probs[1] - (1.0 if label == 1 else 0.0)
            return loss, grad
    
    def predict_gender_proba(self):
        logit = self.forward_gender()
        
        if self.gender_loss_type == 'bce':
            return self.sigmoid(logit)
        else:  # 'nll'
            logits = np.array([0.0, logit])
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            return probs[1]
    
    def forward(self, x_t, training=True):
        self.char_indices.append(x_t)
        
        x_embedded = self.embedding[x_t]
        if training:
            x_embedded = self.dropout(x_embedded, training)

        h_new, cache = self.gru_cell.forward(x_embedded, self.h)
        self.caches.append(cache)

        if training:
            h_new = self.dropout(h_new, training)
        
        self.h = h_new

        logits = self.W_y @ self.h + self.b_y
        return logits
    
    def forward_sequence(self, sequence, training=True):
        logits_list = []
        self.reset_state()
        
        for t in range(len(sequence) - 1):
            logits = self.forward(sequence[t], training)
            logits_list.append(logits)
        
        return logits_list
    
    def backward_letter(self, dL_dy, learning_rate):
        dL_dWy = np.outer(dL_dy, self.h)
        dL_dby = dL_dy
        dL_dh = self.W_y.T @ dL_dy
        
        self.W_y -= learning_rate * dL_dWy
        self.b_y -= learning_rate * dL_dby
        
        dL_dx, dL_dh_prev = self.gru_cell.backward(dL_dh, self.caches[-1], learning_rate)
        
        if self.char_indices:
            x_t_idx = self.char_indices[-1] 
            self.embedding[x_t_idx] -= learning_rate * dL_dx
        
        return dL_dh_prev
    
    def compute_loss(self, sequence, gender_label):
        logits_list = self.forward_sequence(sequence, training=False)
        
        letter_loss = 0
        for t, logits in enumerate(logits_list):
            probs = self.softmax(logits)
            target = sequence[t + 1]
            letter_loss += -np.log(probs[target] + 1e-9)
        
        letter_loss /= len(logits_list) if logits_list else 1
        
        logit = self.forward_gender()
        gender_loss, _ = self.compute_gender_loss(logit, gender_label)
        
        return letter_loss, gender_loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, verbose=True):
        for epoch in range(epochs):
            total_letter_loss = 0
            total_gender_loss = 0
            count = 0
            
            for word, gender_label in zip(X_train, y_train):
                self.reset_state()
                word_loss = 0
                
                hidden_states = []
                all_caches = []
                all_char_indices = []
                
                for t in range(len(word) - 1):
                    x_t = word[t]
                    y_next = word[t + 1]
                    
                    logits = self.forward(x_t, training=True)
                    probs = self.softmax(logits)
                    
                    hidden_states.append(self.h.copy())
                    all_caches.append(self.caches[-1] if self.caches else None)
                    all_char_indices.append(x_t)
                    
                    loss = -np.log(probs[y_next] + 1e-9)
                    word_loss += loss
                    
                    dL_dy = probs.copy()
                    dL_dy[y_next] -= 1
                    self.backward_letter(dL_dy, self.lr)
                
                gender_logit = self.forward_gender()
                
                gender_loss, grad_logit = self.compute_gender_loss(gender_logit, gender_label)
                
                dL_dh = (self.W_gender.T @ np.array([grad_logit])).flatten()
                
                dL_dWgender = np.outer(np.array([grad_logit]), self.h)
                dL_dbgender = grad_logit
                
                self.W_gender -= self.lr * dL_dWgender
                self.b_gender -= self.lr * dL_dbgender
                
                current_caches = self.caches.copy()
                current_char_indices = self.char_indices.copy()
                
                self.caches = all_caches
                self.char_indices = all_char_indices
                
                dL_dh_current = dL_dh
                
                for t in reversed(range(len(hidden_states))):
                    if t < len(self.caches) and self.caches[t] is not None:
                        dL_dx, dL_dh_prev = self.gru_cell.backward(dL_dh_current, self.caches[t], self.lr)
                        dL_dh_current = dL_dh_prev
                        
                        if t < len(self.char_indices):
                            x_t_idx = self.char_indices[t]
                            self.embedding[x_t_idx] -= self.lr * dL_dx
                
                self.caches = current_caches
                self.char_indices = current_char_indices
                self.h = hidden_states[-1] if hidden_states else np.zeros(self.hidden_dim)
                
                total_letter_loss += word_loss / (len(word) - 1) if len(word) > 1 else 0
                total_gender_loss += gender_loss
                count += 1
            
            avg_letter_loss = total_letter_loss / count if count > 0 else 0
            avg_gender_loss = total_gender_loss / count if count > 0 else 0
            
            self.train_loss_history['letter'].append(avg_letter_loss)
            self.train_loss_history['gender'].append(avg_gender_loss)
            
            if X_val is not None and y_val is not None:
                val_letter_loss, val_gender_loss = self._validate(X_val, y_val)
                self.val_loss_history['letter'].append(val_letter_loss)
                self.val_loss_history['gender'].append(val_gender_loss)
                
                if verbose and epoch % 1 == 0:
                    print(f"Epoch {epoch:3d} | Train L: {avg_letter_loss:.4f}, G: {avg_gender_loss:.4f} | "
                        f"Val L: {val_letter_loss:.4f}, G: {val_gender_loss:.4f}")
            else:
                if verbose and epoch % 1 == 0:
                    print(f"Epoch {epoch:3d} | Train L: {avg_letter_loss:.4f}, G: {avg_gender_loss:.4f}")

    def _validate(self, X_val, y_val):
        total_letter_loss = 0
        total_gender_loss = 0
        count = 0
        
        saved_h = self.h.copy()
        saved_caches = self.caches.copy()
        saved_char_indices = self.char_indices.copy()
        
        for word, gender_label in zip(X_val, y_val):
            letter_loss, gender_loss = self.compute_loss(word, gender_label)
            total_letter_loss += letter_loss
            total_gender_loss += gender_loss
            count += 1
        
        self.h = saved_h
        self.caches = saved_caches
        self.char_indices = saved_char_indices
        
        return total_letter_loss / count if count > 0 else 0, total_gender_loss / count if count > 0 else 0
    
    def predict_next_letter(self, char_index, temperature=1.0, reset=True):
        if reset:
            self.reset_state()
        
        logits = self.forward(char_index, training=False)
        
        if temperature < 0.01:
            temperature = 0.01
        
        probs = self.softmax(logits / temperature)
        
        if temperature < 0.1:
            next_idx = np.argmax(probs)
        else:
            next_idx = np.random.choice(len(probs), p=probs)
        
        return next_idx, probs
    
    def generate_name(self, start_char=27, temperature=1.0, end_char=28, max_length=30):
        self.reset_state()
        name_indices = [start_char]
        next_idx = None
        step = 0
        
        while next_idx != end_char and step < max_length:
            next_idx, _ = self.predict_next_letter(name_indices[-1], temperature, reset=False)
            name_indices.append(next_idx)
            step += 1
        
        return name_indices
    
    def generate_multiple_names(self, num_names=10, temperature=1.0):
        names = []
        for i in range(num_names):
            name_indices = self.generate_name(temperature=temperature)
            names.append(name_indices)
        return names
    
    def predict_gender_for_name(self, name_indices):
        """Предсказание пола для имени"""
        self.reset_state()
        for idx in name_indices:
            self.forward(idx, training=False)
        
        return self.predict_gender_proba()
    
    def plot_convergence(self):
        epochs = range(1, len(self.train_loss_history['letter']) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(epochs, self.train_loss_history['letter'], 'b-', label='Train', linewidth=2)
        if self.val_loss_history['letter']:
            axes[0].plot(epochs, self.val_loss_history['letter'], 'r-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('GRU Letter Prediction Loss Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.train_loss_history['gender'], 'b-', label='Train', linewidth=2)
        if self.val_loss_history['gender']:
            axes[1].plot(epochs, self.val_loss_history['gender'], 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'GRU Gender Prediction Loss Convergence ({self.gender_loss_type.upper()})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()