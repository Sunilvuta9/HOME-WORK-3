# Homework 3 — Machine Learning (CS5710)
**University of Central Missouri**  
**Department of Computer Science & Cybersecurity**  
**Course:** CS5710 — Machine Learning  
**Semester:** Fall 2025  

**Student Name:** SUNIL KUMAR VUTA  

---

## 📘 Overview
This repository contains my completed **Home Assignment 3** for the Machine Learning course (CS5710).  
It includes both **Part A (Calculations)** and **Part B (Short Answers)**, covering key concepts of neural networks, perceptrons, SVMs, and PCA.

---

---

## 🧮 Part A — Calculations

### Q1. Multi-Input Feedforward Network
**Inputs:** x₁=2, x₂=1, x₃=3  
**Weights:**  
A = [[0.2, −0.5], [0.1, 0.3], [−0.2, 0.8], [0.4, −0.6]]  
B = [0.7, −1.2, 0.5]  

**Steps:**  
z = Aᵀ·[2,1,3,1] = [0.3,1.1]  
h = σ(z) = [0.5744,0.7503]  
u = 0.7h₁ − 1.2h₂ + 0.5 = 0.0018  
y = σ(u) = 0.5004  

✅ Final: Hidden = [0.5744, 0.7503], Output = 0.5004

---

### Q2. XOR with ReLU Network
Formulas:  
h₁=ReLU(x₁+x₂), h₂=ReLU(x₁+x₂−1), h₃=ReLU(2x₁−x₂)  
y=ReLU(h₁−2h₂+h₃)  

| Input | (h₁,h₂,h₃) | Output y |
|--------|------------|-----------|
| (0,0) | (0,0,0) | 0 |
| (0,1) | (1,0,0) | 1 |
| (1,0) | (1,0,2) | 3 |
| (1,1) | (2,1,1) | 1 |

✅ Not exact XOR — (1,1) should yield 0 but gives 1.

---

### Q3. Decision Boundary & Misclassification
**Decision Rule:** s(x)=x₁−2x₂+1  
**Weights:** w₁=1, w₂=−2, b=1  

Boundary: x₁=2x₂−1  
Normal vector: (1,−2)  

| Point | True Label | s(x) | Pred | Correct? |
|--------|-------------|------|------|-----------|
| (2,1) | 1 | 1 | 1 | ✔ |
| (1,3) | 0 | −4 | 0 | ✔ |
| (3,2) | 1 | 0 | 0 | ✖ |
| (0,1) | 0 | −1 | 0 | ✔ |

✅ Perceptron Loss = 1 (only (3,2) misclassified)

---

### Q4. Multi-Layer Forward Pass (Matrix Style)
Input: (2,3)  
A₁ = [[0.2, −0.3], [0.4, 0.1], [0.5, −0.6]]  
A₂ = [[0.7, −0.5], [−0.2, 0.3], [0.1, 0.4]]  
B = [1.0, −1.2, 0.3]  

Layer 1 → z₁ = [2,3,1]A₁ = [2.1,−0.9], h₁=σ(z₁)=[0.8909,0.2891]  
Layer 2 → h₁⁺=[0.8909,0.2891,1], z₂=h₁⁺A₂=[0.6658,0.0413], h₂=[0.6606,0.5103]  
Output → u=h₂⁺B=0.3465 → y=σ(u)=0.5862  

✅ Final Output: 0.5862

---

### Q5. Linear SVM
Positive: (1,3),(2,2); Negative: (0,0)

Margin equations → b=−1  
w₁=w₂=0.5  

Dual variables: α₁=0, α₂=0.25, α₃=0.25  

✅ Final Hyperplane: 0.5x₁+0.5x₂−1=0 → x₁+x₂=2  
✅ Margin=√2≈1.414, Width=2√2≈2.828

---

### Q6. Nonlinear SVM
Support vectors: s₁=(1,0,y=−1), s₂=(2,1,y=+1)

For ||x||≤2, Φ(x)=(x₁,x₂)  
w=(−1,+1), b=0 → Hyperplane: z₂−z₁=0  

✅ Decision Function: f(x)=−x₁+x₂  
✅ q=(1,1): f=0 → On boundary  
✅ Margin=1/√2≈0.707, Width≈1.414

---

## 🧠 Part B — Short Answers

### Q1. From Biological to Artificial
- **Neuron:** Biological – transmits impulses; Artificial – computational unit.  
- **Synapse:** Biological – junction for signals; Artificial – weighted link.  
- **Activation:** Biological – triggers firing; Artificial – adds nonlinearity.  
- **Nonlinearity:** Enables complex learning, avoids linear collapse.  
- **DAG:** Ensures forward-only signal flow.

---

### Q2. Architecture & Capacity
- **Depth:** Layers between input & output.  
- **Width:** Neurons per layer.  
- **Hidden Unit Roles:** Feature selection, projection.  
- **D < M vs D > M:** Expansion vs compression trade-offs.  
- **Universal Approximation:** One hidden layer can approximate any continuous function.

---

### Q3. Perceptron vs SVM
| Concept | Perceptron | SVM |
|----------|-------------|-----|
| Objective | Separate classes | Maximize margin |
| Optimization | Minimize errors | Maximize gap |
| Generalization | May overfit | Better with noise |
| Example | Simple binary data | Spam email classifier |

---

### Q4. Margins & Support Vectors
- **Margin:** Minimum distance from boundary to points.  
- **Support Vectors:** Define margin & boundary.  
- **Max Margin:** More robustness to noise.

---

### Q5. PCA Concepts
- **Assumption:** Data varies most along main directions.  
- **Definitions:** (1) Max variance projection, (2) Covariance diagonalization.  
- **Centering:** Removes mean bias.  
- **Eigenvalue:** Represents variance of that component.

---

### Q6. PCA vs Random Projection
| Aspect | PCA | Random Projection |
|---------|-----|------------------|
| Basis Choice | By variance | Random |
| Advantage | Meaningful structure | Fast & simple |
| Disadvantage | Expensive, scale-sensitive | Less accurate |

---

## 💡 Key Learnings
- Understood multi-layer feedforward computations.  
- Derived SVM weights and margins analytically.  
- Compared perceptron and SVM optimization.  
- Explored PCA and dimensionality reduction trade-offs.  

---

## ⚙️ How to Run (Optional)
```bash
# Clone the repository
git clone https://github.com/<your-username>/Homework3_ML.git
cd Homework3_ML

# Run numerical code (if implemented)
python3 partA_calculations.py
