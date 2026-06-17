import numpy as np
import matplotlib.pyplot as plt


class LSTMCell:
    """LSTM ячейка, реализованная с нуля"""
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Веса для входного вентиля (input gate - i)
        self.W_i = np.random.randn(hidden_dim, input_dim) * 0.01
        self.U_i = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_i = np.zeros(hidden_dim)
        
        # Веса для забывающего вентиля (forget gate - f)
        self.W_f = np.random.randn(hidden_dim, input_dim) * 0.01
        self.U_f = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_f = np.zeros(hidden_dim)
        
        # Веса для выходного вентиля (output gate - o)
        self.W_o = np.random.randn(hidden_dim, input_dim) * 0.01
        self.U_o = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_o = np.zeros(hidden_dim)
        
        # Веса для ячейки (cell gate - g)
        self.W_g = np.random.randn(hidden_dim, input_dim) * 0.01
        self.U_g = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_g = np.zeros(hidden_dim)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x, h_prev, c_prev):
        i = self.sigmoid(self.W_i @ x + self.U_i @ h_prev + self.b_i)
        f = self.sigmoid(self.W_f @ x + self.U_f @ h_prev + self.b_f)
        o = self.sigmoid(self.W_o @ x + self.U_o @ h_prev + self.b_o)
        g = np.tanh(self.W_g @ x + self.U_g @ h_prev + self.b_g)
        
        c_new = f * c_prev + i * g
        h_new = o * np.tanh(c_new)
        
        cache = {
            'x': x,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'i': i,
            'f': f,
            'o': o,
            'g': g,
            'c_new': c_new,
            'h_new': h_new
        }
        
        return h_new, c_new, cache
    
    def backward(self, dL_dh, dL_dc, cache, learning_rate):
        x = cache['x']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        i = cache['i']
        f = cache['f']
        o = cache['o']
        g = cache['g']
        c_new = cache['c_new']
        
        dL_do = dL_dh * np.tanh(c_new)
        dL_dtanh_c = dL_dh * o
        dL_dc_new = dL_dtanh_c * (1 - np.tanh(c_new)**2) + dL_dc
        
        dL_df = dL_dc_new * c_prev
        dL_dc_prev = dL_dc_new * f
        dL_di = dL_dc_new * g
        dL_dg = dL_dc_new * i
        
        dtanh_g = 1 - g**2
        dL_dg_raw = dL_dg * dtanh_g
        dL_dW_g = np.outer(dL_dg_raw, x)
        dL_dU_g = np.outer(dL_dg_raw, h_prev)
        dL_db_g = dL_dg_raw
        dL_dh_prev_g = self.U_g.T @ dL_dg_raw
        dL_dx_g = self.W_g.T @ dL_dg_raw
        
        d_sigmoid_i = i * (1 - i)
        dL_di_raw = dL_di * d_sigmoid_i
        dL_dW_i = np.outer(dL_di_raw, x)
        dL_dU_i = np.outer(dL_di_raw, h_prev)
        dL_db_i = dL_di_raw
        dL_dh_prev_i = self.U_i.T @ dL_di_raw
        dL_dx_i = self.W_i.T @ dL_di_raw
        
        d_sigmoid_f = f * (1 - f)
        dL_df_raw = dL_df * d_sigmoid_f
        dL_dW_f = np.outer(dL_df_raw, x)
        dL_dU_f = np.outer(dL_df_raw, h_prev)
        dL_db_f = dL_df_raw
        dL_dh_prev_f = self.U_f.T @ dL_df_raw
        dL_dx_f = self.W_f.T @ dL_df_raw
        
        d_sigmoid_o = o * (1 - o)
        dL_do_raw = dL_do * d_sigmoid_o
        dL_dW_o = np.outer(dL_do_raw, x)
        dL_dU_o = np.outer(dL_do_raw, h_prev)
        dL_db_o = dL_do_raw
        dL_dh_prev_o = self.U_o.T @ dL_do_raw
        dL_dx_o = self.W_o.T @ dL_do_raw
        
        self.W_i -= learning_rate * dL_dW_i
        self.U_i -= learning_rate * dL_dU_i
        self.b_i -= learning_rate * dL_db_i
        
        self.W_f -= learning_rate * dL_dW_f
        self.U_f -= learning_rate * dL_dU_f
        self.b_f -= learning_rate * dL_db_f
        
        self.W_o -= learning_rate * dL_dW_o
        self.U_o -= learning_rate * dL_dU_o
        self.b_o -= learning_rate * dL_db_o
        
        self.W_g -= learning_rate * dL_dW_g
        self.U_g -= learning_rate * dL_dU_g
        self.b_g -= learning_rate * dL_db_g
        
        dL_dh_prev = dL_dh_prev_i + dL_dh_prev_f + dL_dh_prev_o + dL_dh_prev_g
        dL_dx = dL_dx_i + dL_dx_f + dL_dx_o + dL_dx_g
        
        return dL_dx, dL_dh_prev, dL_dc_prev


class LSTMRNN:
    
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
        self.lstm_cell = LSTMCell(embedding_dim, hidden_dim)
        
        self.W_y = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_y = np.zeros(vocab_size)
        
        self.W_gender = np.random.randn(1, hidden_dim) * 0.01
        self.b_gender = np.zeros(1)
        
        self.h = np.zeros(hidden_dim)
        self.c = np.zeros(hidden_dim)
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
        self.c = np.zeros(self.hidden_dim)
        self.caches = []
        self.char_indices = []
    
    def forward_gender(self):
        """Возвращает логит (без активации!)"""
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
            
            loss = -np.log(probs[int(label)] + 1e-9)
            
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
        
        h_new, c_new, cache = self.lstm_cell.forward(x_embedded, self.h, self.c)
        self.caches.append(cache)
        
        if training:
            h_new = self.dropout(h_new, training)
        
        self.h = h_new
        self.c = c_new
        
        logits = self.W_y @ self.h + self.b_y
        return logits
    
    def backward_letter(self, dL_dy, learning_rate):
        dL_dWy = np.outer(dL_dy, self.h)
        dL_dby = dL_dy
        dL_dh = self.W_y.T @ dL_dy
        
        self.W_y -= learning_rate * dL_dWy
        self.b_y -= learning_rate * dL_dby
        
        dL_dx, dL_dh_prev, dL_dc_prev = self.lstm_cell.backward(
            dL_dh, np.zeros(self.hidden_dim), self.caches[-1], learning_rate
        )
        
        if self.char_indices:
            x_t_idx = self.char_indices[-1]
            self.embedding[x_t_idx] -= learning_rate * dL_dx
        
        return dL_dh_prev, dL_dc_prev
    
    def compute_loss(self, sequence, gender_label):
        self.reset_state()
        letter_loss = 0
        
        for t in range(len(sequence) - 1):
            logits = self.forward(sequence[t], training=False)
            probs = self.softmax(logits)
            target = sequence[t + 1]
            letter_loss += -np.log(probs[target] + 1e-9)
        
        letter_loss /= len(sequence) - 1 if len(sequence) > 1 else 1
        
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
                
                all_caches = []
                all_char_indices = []
                all_hidden_states = []
                all_cell_states = []
                
                for t in range(len(word) - 1):
                    x_t = word[t]
                    y_next = word[t + 1]
                    
                    logits = self.forward(x_t, training=True)
                    probs = self.softmax(logits)
                    
                    all_caches.append(self.caches[-1] if self.caches else None)
                    all_char_indices.append(x_t)
                    all_hidden_states.append(self.h.copy())
                    all_cell_states.append(self.c.copy())
                    
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
                current_h = self.h.copy()
                current_c = self.c.copy()
                
                self.caches = all_caches
                self.char_indices = all_char_indices
                
                dL_dh_current = dL_dh
                dL_dc_current = np.zeros(self.hidden_dim)
                
                for t in reversed(range(len(all_caches))):
                    if t < len(self.caches) and self.caches[t] is not None:
                        dL_dx, dL_dh_prev, dL_dc_prev = self.lstm_cell.backward(
                            dL_dh_current, dL_dc_current, self.caches[t], self.lr
                        )
                        dL_dh_current = dL_dh_prev
                        dL_dc_current = dL_dc_prev
                        
                        if t < len(self.char_indices):
                            x_t_idx = self.char_indices[t]
                            self.embedding[x_t_idx] -= self.lr * dL_dx
                
                self.caches = current_caches
                self.char_indices = current_char_indices
                self.h = current_h
                self.c = current_c
                
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
        saved_c = self.c.copy()
        saved_caches = self.caches.copy()
        saved_char_indices = self.char_indices.copy()
        
        for word, gender_label in zip(X_val, y_val):
            letter_loss, gender_loss = self.compute_loss(word, gender_label)
            total_letter_loss += letter_loss
            total_gender_loss += gender_loss
            count += 1
        
        self.h = saved_h
        self.c = saved_c
        self.caches = saved_caches
        self.char_indices = saved_char_indices
        
        return total_letter_loss / count, total_gender_loss / count
    
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
        axes[0].set_title('Letter Prediction Loss Convergence (LSTM)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.train_loss_history['gender'], 'b-', label='Train', linewidth=2)
        if self.val_loss_history['gender']:
            axes[1].plot(epochs, self.val_loss_history['gender'], 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title(f'Gender Prediction Loss Convergence - LSTM ({self.gender_loss_type.upper()})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()