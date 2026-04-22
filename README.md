# ArcNet – Swarm-Based Aerial Minefield Navigation

**Robofest 5.0 Finalist**

## Overview

ArcNet is a **swarm intelligence-based aerial robotics system** designed to autonomously guide a human across a simulated minefield using **onboard computation, real-time mapping, and multi-drone coordination**.

This project was developed for **Robofest 5.0 – Aerial Robotics Challenge**, focusing on solving high-risk navigation problems using lightweight UAV swarms.

## Achievements

* 🥇 **AIR 1 (All India Rank 1)** – Ideation Round
* 🚀 **Selected for Grand Finale – Robofest 5.0**
* 💰 **Research Grant Awarded: ₹2,50,000 (GUJCOST)**

## Problem Statement

Design a system of **4 autonomous micro drones (<1 lb each)** that can:

* Navigate a **100-meter minefield within 10 minutes**
* Be controlled using **gestures or voice (no ground station)**
* Detect and map mines using onboard sensing
* Coordinate as a swarm with minimal redundancy
* Generate and communicate a **safe path (1m clearance)**
* Operate with **fully onboard computation**

## Key Contributions
### Autonomous Navigation

* Stable flight using **Pixhawk + PID tuning**
* Guided & stabilize flight modes
* Dynamic path planning using A* Algorithm.

### Human-Drone Interaction

* Gesture-based control
* Real-time visual guidance system
* Image capturing using gestures

### Mine Detection & Mapping

* Detection logic using Yolov7
* Simulated Pi-camera based capture
* Shared digital map across swarm

## Tech Stack

* Pixhawk Flight Controller
* MAVLink / Drone Communication Protocols
* Python (control + scripting)
* Computer Vision (basic pipeline)
* Embedded Systems + Sensor Simulation
* Multi-Agent Coordination Logic

## System Workflow

1. User deploys drone swarm
2. Provides direction via gestures
3. Drones explore and detect mines
4. Swarm builds shared map
5. System computes safe path
6. User is guided across minefield safely

## Impact

* Demonstrates **real-world application of swarm robotics in defense and safety**
* Combines **AI + embedded systems + robotics + human interaction**
* Designed for **high-risk navigation scenarios with minimal infrastructure**

## Patent

**Patent ID:** *20252112067*

## Project Demonstration

* 🎥 **Prototype Demo Video:** *https://drive.google.com/file/d/1lqiLS72ON80JFTtr7gVoXV_X2tuOq_bq/view?usp=drive_link*
* 🛠️ **Build & Development Video:** *https://drive.google.com/file/d/1GLRUoPlaJWJGgiU95KSP2cd1JBWbSQ-3/view?usp=drive_link*
