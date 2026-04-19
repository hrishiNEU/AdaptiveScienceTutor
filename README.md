# 🎓 Adaptive Science Tutor — Reinforcement Learning Agentic System

**Take-Home Final Exam | Hrishikesh Kulkarni | 002340007**  
*Reinforcement Learning for Agentic AI Systems*

---

## Overview

The Adaptive Science Tutor is an agentic AI system that uses reinforcement learning to intelligently select science questions for a student. The agent observes the student's real-time state — knowledge level, fatigue, recent accuracy, and session progress — and decides which question to ask next.

**This is not a system designed to maximise correct answers.** It is an intelligent question-selection agent that learns to pick the most appropriate question for each student at each moment — matching difficulty to knowledge, avoiding repetition, and adapting to fatigue.

---

## Demo

The tutor runs as an interactive multiple-choice web app:

![Quiz Interface](screenshots/demo.png)

- **A/B/C/D answer buttons** with instant feedback
- Correct answer revealed after each response
- Live accuracy, score, and progress tracking
- Per-topic knowledge level bars
- Session summary with final knowledge breakdown

---

## RL Methods Implemented

| Method | Category | Role |
|--------|----------|------|
| **Q-Learning** | Value-Based Learning | Learns a Q-table mapping (state, question) → expected return |
| **UCB Contextual Bandit** | Exploration Strategy | Intelligently selects the best (topic, difficulty) zone per student context |
| **DQN** | Value-Based (Deep) | Neural network Q-function for generalisation across unseen states |

All three agents update simultaneously from the same reward signal after every question.

---

## System Architecture

```
STUDENT STATE  (7-dimensional)
  ├─ topic_idx         [0-9]    Which topic the question covers
  ├─ difficulty        [0-2]    Easy / Medium / Hard
  ├─ acc_bucket        [0-2]    Recent accuracy: struggling / ok / good
  ├─ knowledge_bucket  [0-2]    Topic knowledge: low / mid / high
  ├─ fatigue_bucket    [0-2]    Session fatigue: fresh / moderate / tired
  ├─ session_progress  [0-2]    Early / mid / late in session
  └─ repeat_flag       [0-1]    Has this question been asked before?
           │
           ├──► UCB Bandit ────────► best (topic, difficulty) zone
           │              │
           ├──► Q-Learning ────────► best question within zone
           │
           └──► DQN (7→64→64→500) ► ensemble re-score
                      │
                FINAL ACTION (question 0–499)
                      │
              Student answers (A/B/C/D)
                      │
                REWARD SIGNAL → update all 3 agents
```

---

## Question Bank

**500 questions** across 10 scientific disciplines:

| Topic | Questions |
|-------|-----------|
| Physics | 50 |
| Chemistry | 50 |
| Biology | 50 |
| Astronomy | 50 |
| Earth Science | 50 |
| Mathematics | 50 |
| Computer Science | 50 |
| History of Science | 50 |
| Environmental Science | 50 |
| Human Body | 50 |

Each question includes: question text, correct answer, 3 distractors, difficulty level (easy/medium/hard), and an optional hint.

---

## Results

Trained over **500 episodes × 40 steps** against randomly initialised simulated students:

| Metric | Random Baseline | RL Agent | Improvement |
|--------|----------------|----------|-------------|
| Final Accuracy (last 100 ep) | 15.6% | 25.4% | **+9.8 pp** |
| Final Avg Reward (last 100 ep) | -20.91 | -0.67 | **+20.25** |

### Learning Curves
![Learning Curves](screenshots/fig1_learning_curves.png)

### Early vs Late Training
![Early vs Late](screenshots/fig2_early_vs_late.png)

### DQN Convergence
![DQN Dynamics](screenshots/fig3_dqn_dynamics.png)

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Open `Hrishikesh_Kulkarni_002340007_Take_home_final_exam.ipynb` in [Google Colab](https://colab.research.google.com)
2. Go to **Runtime → Run all**
3. Wait ~2 minutes for training to complete
4. Click the URL printed in the last cell to open the tutor

No installation required — everything runs in Colab.

### Option 2 — Local

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/adaptive-science-tutor
cd adaptive-science-tutor

# Install dependencies
pip install flask numpy matplotlib

# Open the notebook
jupyter notebook Hrishikesh_Kulkarni_002340007_Take_home_final_exam.ipynb
```

Run all cells in order. The last cell will start the Flask server at `http://localhost:5000`.

---

## Repository Structure

```
adaptive-science-tutor/
├── Hrishikesh_Kulkarni_002340007_Take_home_final_exam.ipynb   # Main notebook
├── Hrishikesh_Kulkarni_002340007_Technical_Report.pdf         # Technical report (PDF)
├── Hrishikesh_Kulkarni_002340007_Technical_Report.docx        # Technical report (Word)
├── README.md
└── screenshots/
    ├── demo.png
    ├── fig1_learning_curves.png
    ├── fig2_early_vs_late.png
    ├── fig3_dqn_dynamics.png
    └── fig4_summary.png
```

---

## Technical Details

### State Space
7 dimensions → 3,402 theoretical states (4 × 3 × 3 × 3 × 3 × 3 × 2)

### Reward Function
- Correct answer + difficulty matches knowledge level → **+1.0 to +2.0**
- Correct answer, difficulty mismatched → **+0.4 to +1.4**
- Wrong answer on hard question (struggling student) → **-1.50**
- Repeated question → **-0.80 additional penalty**

### DQN Implementation
Built entirely in **pure NumPy** — no PyTorch or TensorFlow:
- Architecture: `7 → Dense(64, ReLU) → Dense(64, ReLU) → 500`
- Adam optimiser implemented from scratch
- Experience replay buffer: 10,000 transitions
- Target network synced every 100 gradient steps
- He weight initialisation

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Q-Learning α | 0.12 |
| Q-Learning γ | 0.92 |
| Epsilon (start) | 1.0 |
| Epsilon (end) | 0.05 |
| Epsilon decay | 0.997/episode |
| UCB C | 1.4 |
| DQN learning rate | 3e-4 |
| Replay buffer size | 10,000 |
| Batch size | 64 |
| Target update frequency | 100 steps |

---

## Limitations & Future Work

- **No cross-session persistence** — agent weights and student knowledge reset between sessions
- **Simulated students** — trained on synthetic data, not real learner interactions
- **Single environment** — only science tutoring; no multi-domain testing
- **Future**: persistent student models, policy gradient methods (PPO), real student data, larger question bank

---

## Ethical Considerations

- The agent selects questions based on inferred knowledge level — no personal data is stored beyond the active session
- The system is transparent about topic and difficulty for each question
- The reward function explicitly penalises giving hard questions to struggling students to avoid discouragement

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
3. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2), 235–256.
4. VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist*, 46(4), 197–221.
5. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.

---

## License

This project was created for academic purposes as part of a graduate course final exam.
