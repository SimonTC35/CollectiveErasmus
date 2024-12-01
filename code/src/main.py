import numpy as np
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class Simulation:
    def __init__(self, p_dog, p_sheep, p_destination):
        # Constants and initial positions
        self.p_destination = p_destination  # Destination position
        self.radius_operation = 45  # Radius of operation
        self.N = len(p_sheep)  # Number of sheep

        # Initial positions
        self.p_dog = p_dog.astype(np.float32)
        self.p_sheep = p_sheep.astype(np.float32)

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
        self.r_attraction = 400  # Radius of attraction
        self.gamma_a = 450  # Gamma A constant
        self.gamma_b = 375  # Gamma B constant

    def run_step(self):
        """
        Perform one step of the simulation.
        """
        visible_indices = self.visible_sheep()
        herd_center = self.herd_center(visible_indices)

        # Update sheepdog position
        self.update_dog_position(herd_center)

        # Update sheep positions
        self.update_sheep_positions()

        return self.p_dog, self.p_sheep

    def visible_sheep(self):
        """
        Get all sheep visible to the dog in the current step.
        :return: a list of indices of visible sheep.
        """
        vectors = self.p_sheep - self.p_dog
        distances = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        sheep_data = np.array(list(zip(range(len(self.p_sheep)), distances, angles)))
        sheep_data = sheep_data[sheep_data[:, 1].argsort()]

        visible_indices = []
        angle_visibility = {}

        for index, distance, angle in sheep_data:
            if angle not in angle_visibility or angle_visibility[angle] > distance:
                angle_visibility[angle] = distance
                if distance <= self.r_attraction:
                    visible_indices.append(int(index))
        return visible_indices

    def herd_center(self, visible_indices):
        """
        Compute the center of the visible herd.
        """
        if len(visible_indices) > 0:
            visible_positions = self.p_sheep[visible_indices]
            center = np.mean(visible_positions, axis=0)
        else:
            center = None
        return center

    def update_dog_position(self, herd_center):
        """
        Update the position of the sheepdog to guide the herd.
        """
        if herd_center is not None:
            guidance = (herd_center - self.p_destination) / np.linalg.norm(herd_center - self.p_destination)
            cohesion = (self.p_dog - herd_center) / np.linalg.norm(self.p_dog - herd_center)
            control = -self.alpha * guidance + self.beta * cohesion
            self.p_dog += control / np.linalg.norm(control)

    def update_sheep_positions(self):
        """
        Update the positions of the sheep based on the sheepdog's influence.
        """
        for i in range(self.N):
            direction_to_dog = self.p_dog - self.p_sheep[i]
            distance_to_dog = np.linalg.norm(direction_to_dog)

            if distance_to_dog < self.r_attraction:
                force = self.gamma * direction_to_dog / distance_to_dog
            else:
                force = np.zeros(2)

            self.p_sheep[i] += self.alpha_i * force

    def sheep_herd_polygon(self):
        """
        Compute the convex hull of the sheep herd.
        """
        hull = ConvexHull(self.p_sheep)
        return self.p_sheep[hull.vertices]

    def is_converged(self):
        """
        Check if all sheep have reached the destination (within radius r_d).
        """
        distances = np.linalg.norm(self.p_sheep - self.p_destination, axis=1)
        return np.all(distances <= self.r_d)

    def run_simulation(self, max_steps=1000):
        """
        Run the simulation for a given number of steps or until convergence.
        """
        positions = []
        for step in range(max_steps):
            positions.append(self.run_step())
            if self.is_converged():
                print(f"Simulation converged in {step + 1} steps.")
                return positions
        else:
            print("Simulation did not converge.")
        return positions

def animate_simulation(sim, max_steps=100, interval=200):
    """
    Animate the sheepdog simulation.

    :param sim: An instance of the Simulation class.
    :param max_steps: Maximum number of steps to animate.
    :param interval: Delay between frames in milliseconds.
    """
    # Initialize the plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')

    # Set plot limits
    buffer = 10
    min_x = min(np.min(sim.p_sheep[:, 0]), sim.p_dog[0], sim.p_destination[0]) - buffer
    max_x = max(np.max(sim.p_sheep[:, 0]), sim.p_dog[0], sim.p_destination[0]) + buffer
    min_y = min(np.min(sim.p_sheep[:, 1]), sim.p_dog[1], sim.p_destination[1]) - buffer
    max_y = max(np.max(sim.p_sheep[:, 1]), sim.p_dog[1], sim.p_destination[1]) + buffer
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Draw the destination circle
    destination_circle = plt.Circle(sim.p_destination, sim.r_d, color='black', fill=False, linewidth=1.5)
    ax.add_artist(destination_circle)

    # Initialize markers
    sheep_scatter = ax.scatter(sim.p_sheep[:, 0], sim.p_sheep[:, 1], c='blue', label='Sheep')
    dog_marker, = ax.plot(sim.p_dog[0], sim.p_dog[1], 'ro', label='Dog')

    destination_marker, = ax.plot(sim.p_destination[0], sim.p_destination[1], 'ro', label='Dog')
    ax.add_artist(destination_marker)

    # Add legend
    ax.legend(loc='upper right')

    step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update(frame):
        # Run a simulation step
        sim.run_step()

        # Update sheep positions
        sheep_scatter.set_offsets(sim.p_sheep)

        # Update dog position
        dog_marker.set_data([sim.p_dog[0]], [sim.p_dog[1]])

        # Update the step text
        step_text.set_text(f'Step: {frame}')

        return sheep_scatter, dog_marker

    # Create the animation
    anim = FuncAnimation(fig, update, frames=max_steps, interval=interval, blit=False)

    # Display the animation
    plt.show()


if __name__ == "__main__":
    # Initialize the simulation
    paper_dog_position = np.array([63, 10])
    paper_sheep_positions = np.array([
        (57, 50), (44, 71), (59, 64), (73, 71), (78, 60), (82, 71),
        (87, 58), (96, 71), (55, 76), (64, 83), (69, 79), (50, 60),
        (87, 76), (95, 84), (100, 76), (105, 79), (65, 94), (69, 90),
        (105, 85), (79, 95), (84, 90), (90, 99), (100, 55), (105, 60)])
    paper_destination = (115, 300)

    easy_dog_position = np.array([10, 10])
    easy_sheep_positions = np.array([(20, 20), (30, 30), (10, 20), (20, 10)])
    easy_destination = (50, 50)

    # simulation = Simulation(easy_dog_position, easy_sheep_positions,  easy_destination)
    simulation = Simulation(paper_dog_position, paper_sheep_positions,  paper_destination)

    animate_simulation(simulation)
