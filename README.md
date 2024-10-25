# Sheepdog-Driven Algorithm for Sheep Herd Transport

This project explores the **[Sheepdog-Driven Algorithm for Sheep Herd Transport by Liu et al., 2021](https://doi.org/10.23919/CCC52363.2021.9549396)**. The study moves beyond traditional swarm systems, where collective behavior emerges naturally within a group, to address a reverse problem: how a single agent (a "sheepdog") can control and direct a swarm (a "herd of sheep") from one location to another. The objective is to replicate and evaluate this sheepdog-inspired approach, focusing on how a single controlling agent can dynamically influence the herd to accomplish transport tasks.

In this project, we will implement the **backward semi-circle reciprocation algorithm** described in the paper. This algorithm incorporates key behaviors observed in real sheepdogs, including reciprocal movement in a semicircular path and dynamic selection of turn-around points based on the sheep's positions. The "sheep" in this simulation will be individual agents programmed to follow simple, preset rules that govern flocking behavior. The sheepdog, on the other hand, will use these rules strategically to control the sheep’s overall movement toward a defined goal.

After implementing the base algorithm, our team will experiment with its parameters and performance under different conditions. This might include testing variations like obstacle placement, changes in herd size, or adjustments to sheepdog responsiveness. As we analyze the algorithm’s effectiveness in guiding the sheep herd, we may consider developing new behaviors or optimizations to enhance adaptability, with the potential for further applications in fields like autonomous drone or robot control.

## Team Members

- **Idora Ban** - @ind1xa
- **Simon Hehnen** - @SimonTC35
- **Iva Idzojtic** - @viii-debug
- **Lukas Pucher** - @DeinschneeweicherOnkel

## Project Plan

### Starting Point

We will begin by reviewing the Sheepdog-Driven Algorithm paper to understand the methodology, algorithmic details, and problem scope. We’ll implement a simulation environment that mimics the controlled sheep herd environment described in the paper, then gradually build up our understanding by coding a basic prototype of the algorithm.

### Milestones and Timeline

#### First report
- **Tasks**:
  - Review the Sheepdog-Driven Algorithm paper
  - Summarize key concepts
  - Define project objectives
  - Establish basic implementation requirements.
- **Expected Outcome**: A detailed project outline.

#### Second report
- **Tasks**:
  - Implement the algorithm within the simulation environment.
  - Collect initial data and assess the algorithm’s effectiveness, identifying any areas for improvement.
- **Expected Outcome**: An operational simulation with preliminary results and insights for further adjustments.

#### Final report
- **Tasks**:
  - Refine the algorithm based on previous findings.
  - Finalize documentation, summarize results, and analyze the algorithm’s performance across different scenarios.
- **Expected Outcome**: Finalized code, detailed analysis, and a completed project report.
