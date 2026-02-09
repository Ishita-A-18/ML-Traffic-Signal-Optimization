# Hybrid Traffic Signal Control using Vision + Reinforcement Learning



A closed-loop traffic signal control system that combines real-world computer vision, microscopic traffic simulation, and deep reinforcement learning to optimize intersection throughput under partial observability constraints.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technical Approach](#technical-approach)
- [Current Scope](#current-scope)
- [Future Work](#future-work)


## Overview

This project addresses a critical challenge in urban traffic management: **optimizing signal control when complete traffic sensing is unavailable**. In real-world deployments, traffic cameras are often limited to specific approaches, yet controllers must optimize traffic flow across all directions.

Our system implements a **hybrid sensing paradigm** that:
- Uses real video input where available (partial observability)
- Leverages simulation-based estimation for unobserved directions
- Employs Deep Q-Network (DQN) reinforcement learning for adaptive signal timing

This approach bridges the gap between fully simulated control systems and impractical full-camera deployments, making it suitable for real-world applications and research on partial observability in intelligent transportation systems.

---

## Key Features

- **🎥 Hybrid Perception**: Combines real video-based traffic detection with SUMO simulation data
- **🤖 Reinforcement Learning**: DQN agent learns optimal phase-duration policies
- **🔄 Closed-Loop Control**: Real-time feedback system with continuous state-action-reward cycles
- **⚡ Phase-Duration Optimization**: Dynamically adjusts green light duration while maintaining safe phase sequences
- **📊 Realistic Constraints**: Models real-world partial observability scenarios
- **🔧 Modular Design**: Extensible architecture for multi-camera and multi-intersection expansion

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HYBRID TRAFFIC CONTROL SYSTEM               │
└─────────────────────────────────────────────────────────────────┘

         ┌──────────────────┐
         │  Traffic Video   │
         │  (One Approach)  │
         └────────┬─────────┘
                  │
         ┌────────▼─────────┐
         │ Vehicle Detection│
         │   (e.g., YOLO)   │
         └────────┬─────────┘
                  │
         Lane-wise Density
                  │
                  ▼
    ┌─────────────────────────┐         ┌──────────────────┐
    │    Hybrid State         │────────▶│   DQN Agent      │
    │    Aggregation          │         │  (RL Controller) │
    └─────────▲───────────────┘         └────────┬─────────┘
              │                                  │
    ┌─────────┴─────────┐                       │
    │   SUMO Simulator  │              Green Duration
    │  (Other Lanes)    │              Action Selection
    └─────────┬─────────┘                       │
              │                                  │
     Queue Length &                              │
     Waiting Time                                │
              │                                  │
              └──────────────┬───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Traffic Light  │
                    │     Control     │
                    │   (via TraCI)   │
                    └─────────────────┘
```

---

## Technical Approach

### Control Mechanism

The RL agent employs **phase-duration control** rather than phase-selection control:

- **What it controls**: Duration of the current green phase (e.g., 10-60 seconds)
- **What it doesn't control**: Phase sequence or switching logic
- **Why**: Ensures safe signal operation while enabling adaptive optimization

**Action Space**: `[10, 20, 30, 40, 50, 60]` seconds

### State Representation

The system constructs a fixed-length hybrid state vector:

```python
state = [
    video_vehicle_count,      # From computer vision
    video_waiting_time_est,   # Estimated from video
    video_congestion_flag,    # Binary threshold indicator
    
    sumo_lane1_queue,         # From SUMO TraCI
    sumo_lane1_waiting,
    sumo_lane1_count,
    
    sumo_lane2_queue,
    sumo_lane2_waiting,
    sumo_lane2_count,
    
    # ... additional SUMO lanes
]
```

### Reward Function

The agent optimizes for minimal congestion and delay:

```python
reward = -(total_queue_length + total_waiting_time)
```

This encourages:
- Reduced queue lengths across all approaches
- Minimized vehicle waiting times
- Improved overall intersection throughput

### Training Loop

```
1. Observe hybrid state (video + SUMO)
2. Agent selects green duration
3. Execute phase in SUMO simulation
4. Measure queue lengths and waiting times
5. Compute reward
6. Update DQN network
7. Repeat
```

## Current Scope

### ✅ Implemented Features

- SUMO simulation with controllable traffic signals
- TraCI-based real-time signal control interface
- Computer vision-based vehicle detection (single approach)
- Hybrid state construction from heterogeneous sources
- DQN-based phase-duration optimization
- End-to-end closed-loop execution
- Reward computation based on queue and waiting metrics

### 🎯 Design Assumptions

The following assumptions are **intentional** and reflect real-world deployment constraints:

1. **Partial Observability**: Only one approach has real video input
2. **SUMO Approximation**: Unobserved approaches use simulation-based estimation
3. **Fixed Phase Sequence**: Signal progression follows predefined safe patterns
4. **Duration Control**: RL adjusts timing, not phase ordering

## Future Work

### Planned Extensions

- [ ] **Multi-Camera Integration**: Expand to multiple video-monitored approaches
- [ ] **Phase-Specific Actions**: Different action spaces for NS vs. EW phases
- [ ] **Multi-Intersection Coordination**: Network-level optimization
- [ ] **Advanced Vision Models**: Incorporate speed estimation and trajectory prediction
- [ ] **Real-World Calibration**: Validation using recorded traffic datasets
- [ ] **Transfer Learning**: Domain adaptation from simulation to real-world
- [ ] **Safety Constraints**: Incorporate pedestrian and emergency vehicle priority




