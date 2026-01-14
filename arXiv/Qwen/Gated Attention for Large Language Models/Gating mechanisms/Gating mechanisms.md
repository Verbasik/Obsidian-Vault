![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/1.png]]
 
 **Механизм гейтинга** — это способ _управлять потоком информации_ в нейросети через специальные «ворота» (gates), которые решают, **что пропустить, а что приглушить или отфильтровать**.

![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/2.png]]

---

## 1. Интуитивная идея

Представьте, что у вас есть сигнал $h$ (вектор признаков) и «ручка громкости» $g$ для каждого его компонента:

- $g_i \approx 1$ → эту компоненту почти полностью пропускаем;
- $g_i \approx 0$ → почти блокируем;
- $0 < g_i < 1$ → частично ослабляем.

Тогда **гейтинг** — это:

$$  
y = g \odot h  
$$

где

- $h \in \mathbb{R}^d$ — исходный сигнал,
- $g \in [0,1]^d$ — вектор «ворот»,
- $\odot$ — поэлементное умножение,
- $y$ — отфильтрованный сигнал.

**Главная идея:** сеть _сама учится_, какие признаки важны «здесь и сейчас», и усиливает их, а остальные приглушает.

---

## 2. Как строится gate математически

![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/3.png]]

Обычно gate — это **ещё один нейронный слой**, который возвращает вектор в диапазоне $[0,1]$:

$$  
g = \sigma(Wx + b)  
$$

- $x$ — вход (текущий вектор, состояние памяти, контекст и т.п.);
- $W$, $b$ — обучаемые параметры;
- $\sigma$ — сигмоида, поэлементно:  
$$  
\sigma(z) = \frac{1}{1 + e^{-z}} \in (0,1)  
$$

![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/4.png]]

Дальше gate использует либо простое умножение, либо _смешивание_ двух источников:

1. **Маска для одного источника**  

$$  
y = g \odot h  
$$
2. **Смешивание нового и старого состояния** (очень типичный шаблон):

$$  
y = g \odot h_{\text{new}} + (1 - g) \odot h_{\text{old}}  
$$

Это ровно то, что делают LSTM, GRU, Highway-слои, некоторые трансформеры и Mixture-of-Experts.

---

## 3. Мини-пример «на пальцах»

![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/5.png]]

Пусть

- $h = (2, -1, 0.5)$ — вектор признаков,
- gate выдал $g = (0.9, 0.1, 0)$.

Тогда:

$$
y = g \odot h
  = (0.9 \cdot 2,\; 0.1 \cdot (-1),\; 0 \cdot 0.5)
  = (1.8,\; -0.1,\; 0)
$$

Интерпретация:

- первая компонента почти полностью сохраняется;
- вторая сильно ослабляется;
- третья полностью глушится.

---

## 4. Классические примеры гейтинга

### 4.1. LSTM: три (а на деле четыре) gate’а

![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/6.png]]

В LSTM-грубом виде:

- **forget gate** $f_t$: что _забыть_ из старого состояния;
    
- **input gate** $i_t$: что _новое_ записать;
    
- **output gate** $o_t$: какую часть внутреннего состояния показать «наружу».

Формулы (упрощённо):

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

Где:

- $c_t$ — «ячейка памяти»,
    
- gating-вектора $f_t, i_t, o_t \in [0,1]^d$.

Здесь гейтинг буквально решает:

- что оставить из **прошлого** ($f_t$),
    
- что добавить из **нового** ($i_t$),
    
- что отдать **наружу** ($o_t$).

---

### 4.2. GRU: update / reset gate

GRU чуть проще, но идея та же:

![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/7.png]]

- **reset gate** $r_t$ — насколько смотреть на прошлое состояние при вычислении кандидата;
    
- **update gate** $z_t$ — насколько обновить состояние.

$$
\begin{aligned}
&z_t        &&= \sigma(W_z x_t + U_z h_{t-1}) \\
&r_t        &&= \sigma(W_r x_t + U_r h_{t-1}) \\
&\tilde{h}_t&&= \tanh(W_h x_t + U_h (r_t \odot h_{t-1})) \\
&h_t        &&= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$


Вот эта строка:

$$  
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t  
$$

— классический **gated interpolation** между старым и новым состоянием.

---

### 4.3. Highway / Residual с гейтом

![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/8.png]]

Highway layer:

$$
\begin{aligned}
&T(x) &&= \sigma(W_T x + b_T) \quad \text{(transform gate)} \\
&H(x) &&= F(x) \\
&y    &&= T(x) \odot H(x) + (1 - T(x)) \odot x
\end{aligned}
$$


Это то же самое: gate $T(x)$ выбирает, насколько использовать _обработанный_ сигнал $H(x)$, а насколько просто пропустить $x$ напрямую (skip-connection).

Гейтинг здесь помогает:

- стабилизировать обучение;
- сохранить информацию на больших глубинах.

---

### 4.4. Внимание (Attention) как «мягкий гейтинг»

![[arXiv/Qwen/Gated Attention for Large Language Models/Gating mechanisms/Infographics/9.png]]

В self-attention у нас есть набор токенов $x_1, \dots, x_n$. Для токена $i$ attention веса:

$$  
\alpha_{ij} = \text{softmax}_j \left(\frac{q_i^\top k_j}{\sqrt{d}}\right)  
$$

Где $\alpha_{ij} \in (0,1)$, $\sum_j \alpha_{ij} = 1$.

Затем:

$$  
y_i = \sum_{j} \alpha_{ij} v_j  
$$

С точки зрения гейтинга:

- $\alpha_{ij}$ — это **гейт**, который говорит, сколько информации от токена $j$ попадёт в обновлённое представление токена $i$.
    

То есть внимание — это **гейтинг по набору элементов**, а не по компонентам вектора.

---

### 4.5. Mixture-of-Experts (MoE)

![[10.png]]

Есть несколько экспертов $f_1, \dots, f_K$. Гейтер выбирает, к каким экспертом обратиться и с какими весами:

$$  
w = \text{softmax}(W_g x + b_g), \quad w \in [0,1]^K, ; \sum_k w_k = 1,  
$$

$$  
y = \sum_{k=1}^K w_k f_k(x)  
$$

Здесь:

- $w_k$ — **гейт для эксперта $k$**;
    
- может быть «разреженный» гейтинг (top-k), когда реально используются только несколько экспертов.

Это способ _динамически маршрутизировать_ входы к разным подсетям.

---

## 5. Зачем всё это нужно?

![[11.png]]

Гейтинг даёт сразу несколько выгод:

1. **Избирательность информации**  
    Модель учится «внимательно» выбирать, что важно сейчас, а что нет — вместо того чтобы всегда слепо использовать весь вектор.
    
2. **Управление памятью**  
    В рекуррентных сетях (LSTM/GRU) gates буквально решают:
    
    - что забыть,
        
    - что запомнить,
        
    - что выдать на выход.
        
3. **Лучшие градиенты на больших глубинах**  
    Формы вида  
    $$  
    h_t = (1 - g) \odot h_{t-1} + g \odot \tilde{h}_t  
    $$  
    помогают избегать взрыва/затухания градиента, т.к. часть пути остаётся почти  линейной (residual / highway).
    
4. **Динамическая адаптация поведения**  
    MoE, attention, gating в CNN (SE-блоки, GLU и т.п.) — позволяют одной и той же сети **по-разному обрабатывать разные входы**.

---

## 6. Краткое резюме

![[12.png]]

- **Гейтинг** — это механизм, когда сеть _учится строить маску_ $g \in [0, 1]^d$ и модифицирует сигнал через:
    
    - либо $y = g \odot h$,
        
    - либо $y = g \odot h_{\text{new}} + (1-g) \odot h_{\text{old}}$,
        
    - либо веса softmax для набора элементов (attention / MoE).
        
- На практике gates реализуются как маленькая подсеть с сигмоидой или softmax.
    
- Крупные архитектуры, где гейтинг — ключевая идея:
    
    - LSTM / GRU;
        
    - Highway / residual-гейтинг;
        
    - Self-attention и его вариации;
        
    - Mixture-of-Experts и роутеры;
        
    - Канальные гейты в CNN (SE-блоки, GLU и др).
        

![[13.png]]

---
[[Gated Attention for Large Language Models]]