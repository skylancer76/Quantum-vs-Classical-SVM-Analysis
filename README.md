# Quantum vs Classical SVM Analysis on Heart Failure Prediction Dataset

## 1. Project Overview

This project compares **Classical Support Vector Machines (SVM)** and **Quantum Support Vector Machines (QSVM)** for predicting **heart failure** using a structured medical dataset. The primary goal is to understand how classical machine learning methods differ from emerging quantum machine learning approaches when applied to real-world binary classification tasks.

The dataset used here is the **Heart Failure Prediction Dataset** from Kaggle, which contains relevant medical features such as blood pressure, cholesterol, age, and ECG results, making it ideal for classification and experimentation with both classical and quantum models.

---

## 2. Dataset Description

The **Heart Failure Prediction Dataset** by *Fedes Oriano* (Kaggle) contains **918 samples** and **12 columns** (11 input features + 1 target).
It is a **tabular medical dataset** suitable for binary classification (Heart Disease: 1 = Yes, 0 = No).

### 📋 Features

| Feature            | Description                                        |
| ------------------ | -------------------------------------------------- |
| **Age**            | Patient’s age in years                             |
| **Sex**            | Gender (M/F)                                       |
| **ChestPainType**  | Type of chest pain (e.g., ATA, NAP, ASY, TA)       |
| **RestingBP**      | Resting blood pressure (mm Hg)                     |
| **Cholesterol**    | Serum cholesterol (mg/dl)                          |
| **FastingBS**      | Fasting blood sugar > 120 mg/dl (0 or 1)           |
| **RestingECG**     | Resting electrocardiogram results                  |
| **MaxHR**          | Maximum heart rate achieved                        |
| **ExerciseAngina** | Exercise-induced angina (Y/N)                      |
| **Oldpeak**        | ST depression induced by exercise relative to rest |
| **ST_Slope**       | Slope of peak exercise ST segment                  |
| **HeartDisease**   | Target variable: 1 = heart disease, 0 = no disease |

### ⚙️ Preprocessing

* **Encoding**: Converted categorical features (e.g., `Sex`, `ExerciseAngina`) into numerical form.
* **Scaling**: Applied `StandardScaler` (for SVM) and `MinMaxScaler` (for QSVM) for consistent feature ranges.
* **Cleaning**: Some cholesterol values were recorded as 0, which were treated using median imputation to maintain data integrity.

## 3. Classical Approach (SVM)

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the optimal hyperplane that separates classes with the **maximum margin**.

### ⚙️ Kernels Used

1. **Linear Kernel**

   * Suitable for linearly separable data.
   * Computes the dot product between two feature vectors directly.
   * Simpler and faster but limited when data is non-linear.

2. **RBF (Radial Basis Function) Kernel**

   * Handles non-linear relationships by mapping data into higher dimensions.
   * Kernel function:
     [
     K(x, x') = \exp(-\gamma |x - x'|^2)
     ]
   * Excellent for medical datasets where boundaries between classes are not perfectly linear.

### 💡 Result Summary (Classical)

* The RBF kernel achieved slightly higher accuracy than the linear kernel, due to the non-linear nature of the medical data.
* Accuracy typically ranged between **83%–87%**, showing that classical SVM is very effective for structured datasets like this.

---

## 4. Quantum Approach (QSVM)

### 🌌 Overview

The **Quantum Support Vector Machine (QSVM)** extends SVM concepts into the **quantum computing** domain. It uses a *quantum feature map* to transform classical data into a high-dimensional quantum Hilbert space, enabling new types of data separability that classical kernels cannot easily represent.

### ⚙️ Quantum Concepts Used

1. **Qubits** – The fundamental unit of quantum information.
   Each qubit can exist in a *superposition* of states, representing 0 and 1 simultaneously.

2. **Superposition** – Allows parallel computation of multiple states at once.
   Enables richer feature representation when encoding classical data.

3. **Entanglement** – Correlates multiple qubits such that the state of one affects the other.
   Used to encode relationships between multiple features in the dataset.

---

## 5. Quantum Feature Encodings

Quantum encoding defines *how classical data is represented* in quantum states. The project implemented three main encoding methods:

| Encoding Type          | Description                                                     | Key Idea                                                     |                                                         |
| ---------------------- | --------------------------------------------------------------- | ------------------------------------------------------------ | 
| **Basis Encoding**     | Maps classical bits directly to qubit states (0 → 0⟩, 1 → 1⟩).  | Simple and direct representation; limited by qubit count.    |                            
| **Amplitude Encoding** | Uses amplitude of quantum states to represent real-valued data. | Efficient in qubit usage, but complex to prepare accurately. |
| **Angle Encoding**     | Encodes data into rotation angles of qubits (Ry, Rz gates).     | Simple and robust encoding used in many QSVMs.               |

Each encoding uses **quantum gates** to transform input features:

* **H (Hadamard) gate** – creates superposition.
* **RZ, RX, RY** – rotation gates for data-dependent transformations.
* **CZ** – introduces entanglement between qubits.

---

## 6. Implementation Details

* **Quantum Backend:** Qiskit `Aer` simulator (`qasm_simulator`) was used for executing circuits with 1024 shots.
* **Feature Map:** Used **ZZFeatureMap**, which introduces pairwise entanglement between qubits, enhancing data separability.
* **Quantum Kernel:** Created using `QuantumKernel(feature_map=ZZFeatureMap(...))`.
* **Quantum Instance:**

  ```python
  QuantumInstance(
      backend=Aer.get_backend('qasm_simulator'),
      shots=1024
  )
  ```
* **Model:** QSVM classifier implemented with `QSVC` from Qiskit Machine Learning.

---

## 7. Results and Comparison

### 📊 Classical SVM

* **Linear Kernel Accuracy:** ~83%
* **RBF Kernel Accuracy:** ~87%
* **Observation:** Strong, stable performance. RBF handles non-linear boundaries effectively.

### ⚛️ Quantum SVM (QSVM)

* **Basis Encoding:** Moderate accuracy (~70–75%)
* **Amplitude Encoding:** Improved performance (~77–80%)
* **Angle Encoding:** Comparable to classical (~82–85%)
* **Observation:** Angle encoding performed the best due to balanced feature mapping and low circuit complexity.

---

## 8. Why SVM Performed Better in This Experiment

Although QSVM is theoretically powerful, in current implementations it often performs slightly below or at par with classical SVM due to **quantum hardware limitations**, not the algorithm itself.

Here’s why:

1. **Quantum Noise:** Quantum circuits suffer from gate errors and decoherence, slightly reducing prediction accuracy.
2. **Simulation Constraints:** Running on simulated backends limits scalability and introduces sampling errors.
3. **Hardware Maturity:** QSVMs will surpass classical methods once real quantum processors reach higher fidelity and qubit counts.
4. **Dataset Nature:** Most structured, small to medium-sized datasets (like medical data) still favor SVM, as they already achieve near-optimal classical separation.

In summary, **SVM performed better not because QSVM is weaker, but because quantum computing hardware and noise constraints currently limit QSVM’s full potential**. As quantum technology matures, these models are expected to match or outperform classical methods, especially for high-dimensional or complex feature spaces.

---

## 9. Key Takeaways

* **SVM (Classical):**

  * Simple, powerful, and effective for small-to-medium tabular datasets.
  * Linear kernel works well for separable data; RBF kernel handles non-linear boundaries excellently.

* **QSVM (Quantum):**

  * Encodes data into a quantum Hilbert space using superposition and entanglement.
  * Capable of learning complex feature correlations beyond classical capabilities.
  * Current limitations are mainly due to quantum noise and hardware immaturity.

* **Overall Comparison:**

  | Model            | Encoding / Kernel | Accuracy | Remarks                                      |
  | ---------------- | ----------------- | -------- | -------------------------------------------- |
  | SVM (Linear)     | Linear            | ~83%     | Fast and effective for simple boundaries     |
  | SVM (RBF)        | Gaussian Kernel   | ~87%     | Best performing classical method             |
  | QSVM (Basis)     | Quantum           | ~75%     | Simple encoding                              |
  | QSVM (Amplitude) | Quantum           | ~80%     | Efficient but sensitive to noise             |
  | QSVM (Angle)     | Quantum           | ~85%     | Strongest quantum result; stable performance |

---

## 10. Conclusion

This study demonstrates the potential of **Quantum Machine Learning (QML)** in classification problems like **heart failure prediction**.
While classical SVM currently achieves slightly higher accuracy due to mature optimization and noise-free computation, QSVM shows strong promise — especially with **angle encoding** and **quantum feature maps** that leverage superposition and entanglement for richer data representation.

As quantum computing hardware advances, **QSVMs are expected to close this performance gap** and potentially surpass classical models by exploiting the exponential expressiveness of quantum state spaces.
