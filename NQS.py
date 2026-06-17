import numpy as np

class NameQualityMetric:
    
    def __init__(self, real_names, gender_predictor=None):
        self.real_names = real_names
        self.gender_predictor = gender_predictor
        
        self.bigram_probs = self._build_ngram_model(real_names, n=2)
        self.trigram_probs = self._build_ngram_model(real_names, n=3)
        
        self.avg_length = np.mean([len(name) for name in real_names])
        self.common_endings = self._find_common_endings(real_names)
    
    def _build_ngram_model(self, names, n=2):
        from collections import defaultdict, Counter
        
        ngrams = []
        for name in names:
            name_lower = name.lower()
            for i in range(len(name_lower) - n + 1):
                ngrams.append(name_lower[i:i+n])
        
        counter = Counter(ngrams)
        total = sum(counter.values())
        
        probs = defaultdict(lambda: 0.001)  # Laplace smoothing
        for ngram, count in counter.items():
            probs[ngram] = count / total
        
        return probs
    
    def _find_common_endings(self, names):
        from collections import Counter
        
        endings = []
        for name in names:
            if len(name) >= 2:
                endings.append(name[-2:])
            if len(name) >= 3:
                endings.append(name[-3:])
        
        return Counter(endings).most_common(10)
    
    def fluidity_score(self, name):
        if not name:
            return 0
        
        name = name.lower()
        vowels = set('aeiouyаеёиоуыэюя')
        n = len(name)
        
        # 1. Соотношение гласных/согласных
        num_vowels = sum(1 for c in name if c in vowels)
        vowel_ratio = num_vowels / n if n > 0 else 0
        
        # Идеальное соотношение ~0.3-0.5
        if 0.3 <= vowel_ratio <= 0.5:
            vowel_score = 1.0
        else:
            vowel_score = 1 - min(abs(vowel_ratio - 0.4) * 2, 1)
        
        # 2. Штраф за повторяющиеся буквы
        repeats = sum(1 for i in range(n-1) if name[i] == name[i+1])
        repeat_penalty = min(repeats / n, 0.5)
        
        # 3. Штраф за слишком длинные последовательности согласных
        max_consonant_seq = 0
        current = 0
        for c in name:
            if c not in vowels:
                current += 1
                max_consonant_seq = max(max_consonant_seq, current)
            else:
                current = 0
        
        consonant_penalty = max(0, (max_consonant_seq - 3) / 5)
        
        # 4. Бонус за естественные окончания
        ending_bonus = 0
        if len(name) >= 2:
            if name[-2:] in [ending for ending, _ in self.common_endings[:5]]:
                ending_bonus = 0.2
        
        # Итоговая оценка
        raw_score = vowel_score * 0.4 - repeat_penalty * 0.3 - consonant_penalty * 0.3 + ending_bonus
        return max(0, min(1, raw_score))
    
    def plausibility_score(self, name):
        if not name:
            return 0
        
        name = name.lower()
        n = len(name)
        
        # 1. Вероятность биграмм
        bigram_score = 0
        for i in range(n - 1):
            bigram = name[i:i+2]
            bigram_score += self.bigram_probs.get(bigram, 0.001)
        
        if n > 1:
            bigram_score /= (n - 1)
        
        # 2. Вероятность триграмм (для длины > 2)
        trigram_score = 0
        if n > 2:
            for i in range(n - 2):
                trigram = name[i:i+3]
                trigram_score += self.trigram_probs.get(trigram, 0.001)
            trigram_score /= (n - 2)
        else:
            trigram_score = 0.5
        
        # 3. Соответствие длины
        length_score = 1 - min(abs(len(name) - self.avg_length) / 10, 1)
        
        # 4. Специальные правила для имен
        # Имена не должны начинаться с мягкого/твердого знака
        bad_starts = ['ъ', 'ь', 'ы']
        start_penalty = -0.2 if name[0] in bad_starts else 0
        
        # Имена редко имеют более 2 одинаковых букв подряд
        repeat_penalty = 0
        for i in range(n - 2):
            if name[i] == name[i+1] == name[i+2]:
                repeat_penalty = -0.3
                break
        
        # Композитная оценка
        score = (0.4 * bigram_score + 0.3 * trigram_score + 
                 0.2 * length_score + 0.1 * (1 + start_penalty + repeat_penalty))
        
        return max(0, min(1, score))
    
    def diversity_score(self, generated_names):
        if len(generated_names) < 2:
            return 0.5
        
        unique_names = len(set(generated_names))
        uniqueness = unique_names / len(generated_names)
        
        # Разнообразие длин
        lengths = [len(n) for n in generated_names]
        if np.std(lengths) == 0:
            length_div = 0
        else:
            length_div = min(np.std(lengths) / 3, 1)
        
        # Разнообразие первых букв
        first_letters = [n[0] for n in generated_names if n]
        first_letter_div = len(set(first_letters)) / min(len(first_letters), 26)
        
        # Разнообразие последних букв
        last_letters = [n[-1] for n in generated_names if n]
        last_letter_div = len(set(last_letters)) / min(len(last_letters), 26)
        
        score = (0.4 * uniqueness + 0.3 * length_div + 
                 0.2 * first_letter_div + 0.1 * last_letter_div)
        
        return min(1, score)
    
    def gender_consistency_score(self, generated_names):
        if not self.gender_predictor or len(generated_names) < 2:
            return 0.5
        
        try:
            genders = [self.gender_predictor(name) for name in generated_names]
            male_ratio = sum(genders) / len(genders)
            
            # Имена должны быть примерно 50/50
            if 0.4 <= male_ratio <= 0.6:
                return 1.0
            else:
                return 1 - min(abs(male_ratio - 0.5) * 2, 1)
        except:
            return 0.5
    
    def compute(self, generated_names):
        if not generated_names:
            return 0, {}
        
        fluidity = np.mean([self.fluidity_score(name) for name in generated_names])
        plausibility = np.mean([self.plausibility_score(name) for name in generated_names])
        diversity = self.diversity_score(generated_names)
        gender_consistency = self.gender_consistency_score(generated_names)
        
        weights = {
            'fluidity': 0.4,
            'plausibility': 0.3,
            'diversity': 0.2,
            'gender_consistency': 0.1
        }
        
        total_score = (weights['fluidity'] * fluidity + 
                       weights['plausibility'] * plausibility + 
                       weights['diversity'] * diversity + 
                       weights['gender_consistency'] * gender_consistency)
        
        components = {
            'fluidity': fluidity,
            'plausibility': plausibility,
            'diversity': diversity,
            'gender_consistency': gender_consistency
        }
        
        return total_score, components
    
    def detailed_report(self, generated_names, model_name="Model"):
        total_score, components = self.compute(generated_names)
        
        print(f"Name Quality Score - {model_name}")
        print(f"Total Score: {total_score:.3f} / 1.000")
        print(f"\nComponent Scores:")
        print(f"  Fluidity:       {components['fluidity']:.3f} (weight: 0.4)")
        print(f"  Plausibility: {components['plausibility']:.3f} (weight: 0.3)")
        print(f"  Diversity:   {components['diversity']:.3f} (weight: 0.2)")
        print(f"  Gender Consistency:    {components['gender_consistency']:.3f} (weight: 0.1)")
