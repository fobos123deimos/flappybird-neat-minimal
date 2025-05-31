# 🧠 Flappy-NEAT-Minimal

This repository implements the project presented in the paper:

> 📄 [**A Minimal Training Strategy to Play Flappy Bird Indefinitely with NEAT**](https://www.sbgames.org/sbgames2019/files/papers/ComputacaoFull/198468.pdf)

It introduces a **minimal yet effective training strategy** using the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm to evolve a neural agent capable of playing Flappy Bird indefinitely. The core idea lies in simplifying the learning environment through **controlled scenario generation**, enabling the NEAT algorithm to converge more efficiently.

---

🎥 [Watch a sample of the trained NEAT agent playing Flappy Bird (MP4)](assets/demo_result.mp4)

---

## 🧠 Dependencies & Libraries

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![NEAT-Python](https://img.shields.io/badge/neat--python-0.92-blue?style=flat-square)](https://neat-python.readthedocs.io/en/latest/) [![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/) [![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.0-11557C?style=flat-square&logo=matplotlib&logoColor=white)](https://matplotlib.org/) [![Graphviz](https://img.shields.io/badge/Graphviz-2.50+-E10098?style=flat-square&logo=graphviz&logoColor=white)](https://graphviz.org/) [![PyQt5](https://img.shields.io/badge/PyQt5-5.15.9-41CD52?style=flat-square&logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/) [![PLE](https://img.shields.io/badge/PLE-Custom--Modified-orange?style=flat-square)](https://github.com/ntasfi/PyGame-Learning-Environment)

### ✅ Main Usage per File:

| File                                | Libraries Used                                                     |
|-------------------------------------|--------------------------------------------------------------------|
| `scr/train_agent.py`                | neat-python, numpy, ple_custom                                     |
| `scr/evaluate_agent.py`            | neat-python, ple_custom, PyQt5                                     |
| `scr/neat_visualizations.py`       | matplotlib, graphviz, numpy                                        |
| `scr/score_display_window.py`      | PyQt5                                                              |

---

## 📘 Mathematical & Computational Foundations

### 🧠 NEAT Algorithm

NEAT evolves both weights and topology of neural networks by starting with minimal structures and progressively adding complexity. Key mechanisms include:

- **Speciation** based on genomic distance:

  $$
  \delta = c_1 \cdot \frac{E}{N} + c_2 \cdot \frac{D}{N} + c_3 \cdot \bar{W}
  $$

  Where:
  - $E$: number of excess genes  
  - $D$: number of disjoint genes  
  - $\bar{W}$: average weight difference of matching genes  
  - $N$: normalization factor  
  - $c_1, c_2, c_3$: speciation coefficients

- **Fitness Sharing** and **Historical Markings** for safe crossover  
- **Mutation operators**: adding/removing nodes and connections

References:
- Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies*, ECJ 2002  
- Floreano et al., *Neuroevolution: from Architectures to Learning*, 2008

---

### 🎮 Game Domain: Flappy Bird

The agent is evolved to play Flappy Bird, receiving 3 inputs:
- Player vertical position ($y_p$)
- Bottom pipe Y ($y_b$)
- Gap center Y: $y_c = \frac{y_t + y_b}{2}$

And producing one output:
- Jump if output $\geq 0.4$

---

### 🧮 Custom Fitness Function

For each scenario $s_i$:

$$
F_i = w_1 \cdot \frac{P}{195} + w_2 \cdot \frac{S}{3} - w_3 \cdot \frac{|y_p - y_c|}{\text{height}}
$$

Where:
- $P$: number of frames survived  
- $S$: number of pipes passed  
- $y_p$: player vertical position  
- $y_c$: center of pipe gap  

Final fitness across 3 scenarios:

$$
F = \frac{F_1 \cdot p_1 + F_2 \cdot p_2 + F_3 \cdot p_3}{p_1 + p_2 + p_3}
$$

This approach blends behavioral performance and position precision, encouraging generalization.

---

## 🎮 About the PLE (PyGame Learning Environment)

The [PLE](https://github.com/ntasfi/PyGame-Learning-Environment) provides an interface for reinforcement learning agents to interact with classic arcade-style games. This project uses a **custom modified version** located in:

```
ple_custom/
```

Key changes:
- Control over pipe-gap positioning  
- Deterministic scenario generation for consistent evolution  

All modifications are documented in `ple_custom/NOTICE.txt`.

---

## 📂 Repository Structure

| Path                                 | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| `assets/demo_result.mp4`             | Sample of a fully trained NEAT agent                                       |
| `config/flappy_neat_feedforward_config` | NEAT configuration file (feedforward net, 3 inputs, 1 output)          |
| `ple_custom/`                        | Custom version of PLE (modularized and adapted)                            |
| `scr/train_agent.py`                | Evolves population using NEAT                                              |
| `scr/evaluate_agent.py`             | Loads best genome and shows it playing with GUI                            |
| `scr/neat_visualizations.py`        | Plots fitness over generations and speciation curves                       |
| `scr/score_display_window.py`       | GUI component to display current score                                     |
| `LICENSE`                            | MIT License                                                                |
| `ple_custom/NOTICE.txt`             | Describes legal and authorship of modified PLE                            |

---

## ▶️ How to Run

### 🔧 Installation

```bash
git clone https://github.com/youruser/FLAPPYBIRD-NEAT-MINIMAL.git
cd FLAPPYBIRD-NEAT-MINIMAL
pip install -r requirements.txt
```

You may also need:

```bash
sudo apt install graphviz
```

---

### 🚀 Training the Agent

```bash
python scr/train_agent.py
```

- Trains the population using NEAT  
- Visualizes average/best fitness over time  

---

### 🎯 Evaluating the Agent

```bash
python scr/evaluate_agent.py
```

- Loads the best genome from checkpoint  
- Displays game and current score using PyQt5  

---

## 📄 Third-Party Notice

This repository includes a modified version of the [PyGame Learning Environment (PLE)](https://github.com/ntasfi/PyGame-Learning-Environment).

> All changes are described in `ple_custom/NOTICE.txt`.

---

## 🪪 License

MIT License – see `LICENSE`.
