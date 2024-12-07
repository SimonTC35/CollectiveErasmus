# Report 1

## Abstract
TODO abstract 

## Introduction
This project explores the Sheepdog-Driven Algorithm for Sheep Herd Transport by Liu et al., 2021 [TODO cite]. The study moves beyond traditional swarm systems, where collective behavior emerges naturally within a group, to address a reverse problem: how a single agent (a "sheepdog") can control and direct a swarm (a "herd of sheep") from one location to another. The objective is to replicate and evaluate this sheepdog-inspired approach, focusing on how a single controlling agent can dynamically influence the herd to accomplish transport tasks.

The initial phase of this project involves implementing the algorithm described in the paper using Python. Through this implementation, we will conduct tests and fine-tune parameters to evaluate the effectiveness and adaptability of the model as outlined in the study. In subsequent phases, we will introduce several extensions, including the addition of environmental obstacles, the presence of outlier sheep, and variations in the characteristics of the controlling agent.

To effectively present our findings, we will generate visual representations using Unity. The Python-based algorithm will compute the movement paths for each sheep and the sheepdog, which will then be visualized. These visualizations will be produced post-simulation rather than in real time.

Our results will be analyzed in the context of related studies, and we will conclude with a summary of our findings, including any insights gained from the extensions introduced.

## The Algorithm
The algorithm defines the sheepdog’s movement as a backward semi-circular trajectory, allowing it to drive the sheep herd from behind [TODO cite]. The sheepdog operates within a two-dimensional plane, using an xx-yy coordinate system to track both its own location and that of each sheep. The sheepdog aims to guide the herd toward a designated target area by pushing them forward in a controlled manner.

Key aspects of the algorithm include the sheepdog's field of view, which restricts its awareness to only those sheep within its line of sight. Consequently, the sheepdog's response is dynamically adjusted based on the position and behavior of visible sheep, meaning it does not always have complete information about the entire herd.

[TODO insert picture from paper with dogs field of view]

The control process of the sheepdog is divided into two phases:

1. Initialization: This phase involves setting key design parameters such as viewing angles, approach distances, and thresholds for direction adjustments. A time limit is also set for the operation.

2. Iteration: In the main phase, the algorithm evaluates the position of the sheep relative to the target and adjusts the sheepdog's movement accordingly. The sheepdog decides whether to move directly toward the herd or to take a detour based on the herd's overall position. When detouring, it aligns itself with the rightmost or leftmost visible sheep, adjusting its angle to efficiently steer the herd without unnecessary deviation from the target path.

The algorithm operates through a series of conditions to ensure energy efficiency and maintain herd cohesion, continuing until all sheep are guided to the target area or the time limit is reached.

[TODO insert pseudo code from the paper]

## Related Work
The following papers serve as related work.

1. Reynolds, C. W. (1987). "Flocks, Herds, and Schools: A Distributed Behavioral Model."
This foundational paper introduced a model for simulating collective animal behavior, particularly in flocks, herds, and schools, using local rules that individuals follow to achieve realistic swarm behaviors. Reynolds' work on "boids" (bird-like objects) has influenced many swarm control and robotics algorithms and offers foundational concepts that relate to herding behaviors in your project.

2. Strömbom, D., et al. (2014). "Solving the shepherding problem: heuristics for herding autonomous, interacting agents." Journal of the Royal Society Interface, 11(100).
This study examines the problem of herding a group of autonomous agents using a single shepherding agent. It presents several heuristics and strategies that are relevant to understanding and improving single-agent control over a group, making it a useful comparative study for your project.

3. Lien, J.-M., & Pratt, E. (2009). "Interactive Herding of a Group of Autonomous Agents Using a Single Agent." Proceedings of the 2009 Symposium on Interactive 3D Graphics and Games.
This paper explores how a single controlling agent can guide a group of autonomous agents, with a focus on strategies that maintain group cohesion and manage obstacles. It has practical applications in robotics and provides valuable insights into interactive and real-time herding control.

4. Bayazit, O. B., Lien, J.-M., & Amato, N. M. (2002). "Better Group Behaviors in Complex Environments Using Global Roadmaps." Proceedings of the 8th International Conference on Artificial Intelligence Planning and Scheduling.
This paper presents methods for guiding groups through complex environments using precomputed roadmaps. It emphasizes control and navigation in challenging environments with obstacles, which is relevant for your project's potential extensions involving obstacles and other environmental factors.

These papers provide a broad foundation for the core concepts in swarm control, herding algorithms, and multi-agent systems relevant to your project. They will allow you to discuss how the Sheepdog-Driven Algorithm aligns with or differs from other approaches in terms of strategy, efficiency, and adaptability.

## Methodology
The practical component of this project is divided into two primary tasks: implementing the simulation algorithm in Python and developing a visualization system in Unity using C++. The methodology section outlines the approach taken to develop these components, along with the processes for testing, refining, and analyzing results.

### Simmulation
In the simulation phase, the Sheepdog-Driven Algorithm, will be implemented in Python to replicate the sheepdog-herd dynamics on a two-dimensional plane. The simulation aims to model the movements and interactions of both the sheepdog and the sheep, capturing the complex behaviors that arise as the sheepdog attempts to guide the herd to a target location. In the original paper, this was done with MATLAB.

1. Algorithm Implementation: The algorithm will be programmed to include the sheepdog’s backward semi-circular motion, a key mechanism for maintaining control over the herd. This involves defining functions that represent the sheepdog’s movements, perception field, and the reactive behaviors of individual sheep.

2. Parameter Tuning: To evaluate and optimize the algorithm’s effectiveness, parameters such as the sheepdog’s field of view, speed, and interaction range with sheep will be adjustable. This parameter tuning will allow us to observe how changes in these factors influence the sheepdog’s ability to manage the herd under various scenarios.

3. Test Scenarios: Various test cases will be created to examine the algorithm’s performance under different conditions, including:
    - Basic Movement: Verifying the algorithm’s ability to guide the herd in open space.
    - Extended Dynamics: Introducing obstacles and outlier sheep to assess how the algorithm adapts to environmental complexity and group inconsistencies.
    - Different Agent Types: Modifying attributes such as the sheepdog’s behavior, speed, or strategy to observe effects on control and cohesion within the herd.

This Python-based simulation will output path data for each sheep and the sheepdog, which will be saved and used as input for visualization in Unity.

### Visualisation
The visualization component of this project is developed in Unity using C++ to render the movement data generated from the Python simulation, providing an interactive and comprehensible format for analyzing the algorithm's behavior. This allows for a detailed observation of how effectively the sheepdog can guide the herd under various conditions.

The visualization process begins with importing path data from the Python simulation into Unity. This data represents each agent’s movements, enabling us to create animated paths that reflect the interactions between the sheepdog and sheep over time. A virtual environment is then constructed in Unity to provide a realistic setting for the simulation. This environment includes basic elements such as ground planes, obstacles, and boundaries, mirroring the conditions modeled in the Python simulation to enhance visual coherence and aid in understanding how environmental factors affect the sheepdog’s control over the herd.

Unity’s visualization setup focuses on representing the dynamic interactions between the sheepdog and the herd. Various visualization techniques, including path trails, speed indicators, and field-of-view displays, are used to highlight important aspects of the behavior. This design provides insights into how the algorithm reacts to elements like obstacles, outliers, and different herd dynamics.

As the visualization is not rendered in real-time, the animation will be recorded and played back in segments. This approach enables static analysis, where we can pause and closely examine specific interactions or adaptations within the algorithm. By allowing for playback and frame-by-frame inspection, the Unity-based visualization provides a valuable tool for evaluating the algorithm’s effectiveness, adaptability, and overall performance in guiding the herd through diverse scenarios.

## Results and Discussion
After running multiple visualizations and tests, we are able to evaluate the findings from the initial paper. This evaluation is done by comparing the changes we made to the algorithm and setup with the cases reproduced from the paper.

We then place our project in the larger context of collective behavior. We discuss how our findings might impact research in this area and contribute to real-world applications.

We conclude the paper by providing an outlook on the future. What are the next steps? How could our findings be applied outside the lab? Do we now have the ideal sheepdog? Are there any further improvements or new directions we should explore?