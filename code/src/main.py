import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


class Simulation:
    def __init__(self, p_dog, p_sheep, p_destination):
        # Constants and initial positions
        self.p_destination = p_destination # Destination position
        self.radius_operation = 45  # Radius of operation
        self.N = 24  # Number of sheep

        # Initial positions
        self.p_dog = p_dog # Initial position of sheepdog
        self.p_sheep = p_sheep

        # Parameters
        self.alpha_i = 0.1  # Constant a for agents
        self.omega_i = 0.1  # Constant ω for agents
        self.alpha = 7000  # Alpha constant
        self.beta = 1400  # Beta constant
        self.gamma = -140  # Gamma constant
        self.r_n = 50  # Neighbor radius
        self.r_s = 5  # Safety radius
        self.r_r = 15  # Repulsion radius
        self.r_g = 20  # Goal radius
        self.r_d = 30  # Destination radius
        self.phi_t = 2 * np.pi / 3  # Target angle
        self.phi_l = -4  # Left angle limit
        self.phi_r = 4  # Right angle limit
        self.r_attraction = 40  # Radius of attraction
        self.gamma_a = 450  # Gamma A constant
        self.gamma_b = 375  # Gamma B constant

    def run_step(self):
        return self.p_dog, [sheep_position for sheep_position in self.p_sheep]


def animate(positions, destiantion, radius, interval=200):
    """
    Animate the movement of the sheepdog, sheep positions, and target over time.

    Args:
    - positions: A list of [sheepdog_position, [sheep_positions]].
      - sheepdog_position: Current position of the sheepdog as a tuple (x, y).
      - sheep_positions: Current positions of all sheep as a list of (x, y) tuples.
    - target: Target position as a tuple (x, y).
    - radius: Radius around the target to be displayed.
    - interval: Time interval in milliseconds between frames (default is 200ms).
    """
    # Calculate dynamic plot limits
    all_positions = np.array([
        [sheepdog_position, *sheep_positions]
        for sheepdog_position, sheep_positions in positions
    ])
    all_positions_flat = all_positions.reshape(-1, 2)
    x_min, y_min = all_positions_flat.min(axis=0) - 10  # Add margin
    x_max, y_max = all_positions_flat.max(axis=0) + 10  # Add margin

    # Include target position in limits
    x_min = min(x_min, destiantion[0] - radius - 10)
    x_max = max(x_max, destiantion[0] + radius + 10)
    y_min = min(y_min, destiantion[1] - radius - 10)
    y_max = max(y_max, destiantion[1] + radius + 10)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min(x_min, y_min), max(x_max, y_max))
    ax.set_ylim(min(x_min, y_min), max(x_max, y_max))

    # Create scatter plots for sheep, sheepdog, and the target
    sheep_plot, = ax.plot([], [], 'bo', markersize=8, label="Sheep")  # Blue dots for sheep
    sheepdog_plot, = ax.plot([], [], 'ro', markersize=8, label="Sheepdog")  # Red dot for sheepdog
    target_plot, = ax.plot(destiantion[0], destiantion[1], 'k*', markersize=8, label="Sheepfold")  # Green cross for target

    # Add a circle to represent the target radius
    target_circle = Circle(destiantion, radius, color='red', fill=False, linestyle='--')
    ax.add_patch(target_circle)

    # Initialize the plot data (empty)
    def init():
        sheep_plot.set_data([], [])
        sheepdog_plot.set_data([], [])
        return sheep_plot, sheepdog_plot, target_plot, target_circle

    # Update function for animation
    def update(frame):
        # Extract the current sheepdog position and sheep positions for the current frame
        sheepdog_position, sheep_positions = positions[frame]

        # Update the sheep plot data (x, y positions of sheep)
        sheep_plot.set_data(np.array(sheep_positions)[:, 0], np.array(sheep_positions)[:, 1])

        # Update the sheepdog plot data (x, y position of sheepdog)
        sheepdog_plot.set_data([sheepdog_position[0]], [sheepdog_position[1]])

        return sheep_plot, sheepdog_plot, target_plot, target_circle

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(positions), init_func=init,
                        blit=True, interval=interval)

    # Add labels and a legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Sheepdog and Sheep Movement with Target")
    ax.legend(loc="upper left")

    # Show the animation
    plt.show()

if __name__ == "__main__":
    # Initialize the simulation
    simulation = Simulation(np.array([63, 10]), np.array([
            (57, 50), (44, 71), (59, 64), (73, 71), (78, 60), (82, 71),
            (87, 58), (96, 71), (55, 76), (64, 83), (69, 79), (50, 60),
            (87, 76), (95, 84), (100, 76), (105, 79), (65, 94), (69, 90),
            (105, 85), (79, 95), (84, 90), (90, 99), (100, 55), (105, 60)]), (115, 300) )

    positions = [] # Array with [sheepdog_position, [sheep_positions]] for later animation

    # Run a few steps of the simulation
    for step in range(10):
        positions.append(simulation.run_step())

    # animate the calculated paths
    animate(positions, simulation.p_destination, simulation.r_d)

