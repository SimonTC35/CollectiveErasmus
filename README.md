# Sheepdog-Driven Algorithm for Sheep Herd Transport

This project explores the **[Sheepdog-Driven Algorithm for Sheep Herd Transport by Liu et al., 2021](https://doi.org/10.23919/CCC52363.2021.9549396)** which focuses on how a single agent (a "sheepdog") can guide and control a group (the "herd of sheep") to move from one location to another. The goal of this project was to replicate and evaluate this approach, specifically implementing the backward semi-circle reciprocation algorithm described in the paper.

## Team Members

- **Idora Ban** - @ind1xa
- **Simon Hehnen** - @SimonTC35
- **Iva Idzojtic** - @viii-debug
- **Lukas Pucher** - @DeinschneeweicherOnkel

## Results

A simulation based on the algorithm was developed, modeling the interactions between the sheepdog and the sheep using local interaction rules. Movement dynamics for both the sheep and sheepdog were defined, incorporating behaviors like attraction, repulsion, and visibility checks. Real-time position updates were implemented to simulate their behavior.

Additionally, a 3D visualization was created to help illustrate the dynamic interactions between the sheepdog and the herd, providing a clearer understanding of their movement patterns and the algorithm’s behavior.

The plan was to experiment with various parameters and evaluate the algorithm’s performance under different conditions after implementing the base algorithm. However, while the simulation followed the methodology outlined in the paper, the herd was not consistently guided to the designated goal. The sheep’s movement responded to the sheepdog, but the dog was unable to lead the herd to the final destination as expected. These discrepancies suggest that further refinement or adjustments may be needed.

## Instructions

### Step 1: Set up Unity Project
1. Create a new **Empty 3D project** in Unity.
2. **Import the Unity package** (`.unitypackage`) provided with the project. This package includes all necessary assets and scripts for visualization.

### Step 2: Import Textual Instructions (Optional)
1. If you wish to include textual instructions, import the **TextMeshPro (TMP)** package in Unity.
2. Ensure the appropriate TextMeshPro settings are configured to display the instructions correctly.

### Step 3: Prepare Python Code
1. Ensure that Python is installed on your system and the required libraries are available.
2. Open the Python script that generates the movement data.
3. Modify the following variables in the script to point to the correct directories:
   - `EXTRACT_COORDS`: Set this flag to `True` to extract coordinates.
   - `PATH_TO_UNITY_PROJECT`: Provide the path to your Unity project folder.

### Step 4: Run the Python Simulation
1. Execute the Python script. It will generate and save the `.json` files that contain the coordinates of the sheep and sheepdog entities during the simulation.
2. These `.json` files are automatically updated as the simulation progresses.

### Step 5: Run the Simulation
1. Press the **Play** button in Unity to start the simulation.
2. Watch the sheepdog and the sheep interact according to the algorithm's dynamics. The visualization will update based on the movement data provided by the Python simulation.
3. During runtime, it is possible to modify simulation parameters (speed, current step, visualize paths...) through the gameobject Manager
