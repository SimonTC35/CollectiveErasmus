import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull


class Simulation:
    def __init__(self, q, p, p_d):
        # Constants and initial positions
        self.p_d = p_d  # Destination position
        self.rho_0 = 45
        self.rho_v = 50 # vision radius of dog
        self.N = len(p)  # Number of sheep
        self.T = 1 # sampling period


        self.k = 0

        # Initial positions
        self.q = q.astype(np.float32) # dog position
        self.p = p.astype(np.float32) # sheep position

        # Parameters
        self.alpha_i = 0.1  # Constant a for agents
        self.omega_i = 0.1  # Constant ω for agents
        self.alpha = 7000  # Alpha constant
        self.beta = 1400  # Beta constant
        self.gamma = -140  # Gamma constant
        self.rho_n = 50  # Neighbor radius
        self.rho_s = 5  # Safety radius
        self.rho_r = 15  # Repulsion radius
        self.rho_g = 20  # Goal radius
        self.rho_d = 30  # Destination radius
        self.phi_t = 2 * np.pi / 3  # Target angle
        self.phi_l = -4  # Left angle limit
        self.phi_r = 4  # Right angle limit
        self.r_a = 40  # Radius of attraction
        self.gamma_a = 450  # Gamma A constant
        self.gamma_b = 375  # Gamma B constant

        self.theta_t = self.alpha_i * (np.pi / 180) * np.sin(self.omega_i * self.k * self.T)
        self.theta_r = 1
        self.theta_l = 1

        self.lamda_k = 1


    def run_simulation(self, max_steps=1000):
        """
        Run the simulation for a given number of steps or until convergence.
        In the paper algo it is everything until line 6
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

    def run_step(self):
        """
        Perform one step of the simulation.
        In the paper algorithm this is line 6
        """
        p_qi = self.p - self.q
        p_di = self.p - self.p_d


        V = self.visible_sheep() # all shepp visible from the dogs POV
        # herd_center = self.herd_center(V) # herd center of the visible sheep

        P_s = self.sheep_herd_polygon() # the sheep herd polygon (a convex hull with sheep as verticies)


        D_cd = (self.p_d - self.p) / np.linalg.norm(self.p_d - self.p)
        D_qd = (self.p_d - self.q) / np.linalg.norm(self.p_d - self.q)

        D_l, D_r = self.left_most_right_most_sheep(V, self.q) # left most and right most sheep from the dogs POV
        C_l, C_r = self.left_most_right_most_sheep(V, self.p_d) # left most and right most sheep from the destinations POV

        Q_l, Q_r = self.left_right_set()

        L_c = cosine_sim(D_cd, self.q-C_r)
        R_c = cosine_sim(D_cd, self.q-C_l)



        # if wee have all variables, we simply can finish the algo and update
        # the positions step by step

        u_k = 0
        if np.allclose(Q_l, self.q) and L_c > self.theta_t:
            self.lambda_k = 0
            if np.linalg.norm(self.q - D_r) >= self.r_a:
                u_k = self.gamma_a*((self.q - D_r) / np.linalg.norm(self.q - D_r))
            else:
                u_k = self.gamma_b*rotation_matrix(self.theta_r)*((self.q - D_r) / np.linalg.norm(self.q - D_r))
        elif (np.allclose(Q_r, self.q) and R_c > self.theta_t) or self.lamda_k == 1:
            self.lambda_k = 1
            if np.linalg.norm(self.q - D_l) >= self.r_a:
                u_k = self.gamma_a*((self.q - D_l) / np.linalg.norm(self.q - D_l))
            else:
                u_k = self.gamma_b*rotation_matrix(self.theta_l)*((self.q - D_l) / np.linalg.norm(self.q - D_l))
        else:
            if np.linalg.norm(self.q - D_r) >= self.r_a:
                u_k = self.gamma_a*((self.q - D_r) / np.linalg.norm(self.q - D_r))
            else:
                u_k = self.gamma_b*rotation_matrix(self.theta_r)*((self.q - D_r) / np.linalg.norm(self.q - D_r))

        self.q = self.q + self.T*u_k


        self.k += 1
        return self.q, self.p

    def left_right_set(self):
        Q_l = []
        Q_r = []

        # Loop through all sheep positions
        for sheep_position in self.p:
            # Compute the vector pd - x (in this case sheepfold - sheep position)
            direction_vector = self.p_d - sheep_position

            # Normalize the direction vector
            if np.linalg.norm(direction_vector) == 0:
                continue  # Avoid division by zero if vectors coincide
            unit_vector = direction_vector / np.linalg.norm(direction_vector)

            # Calculate angle with respect to x-axis
            angle = np.arctan2(unit_vector[1], unit_vector[0])  # Angle in radians

            # Check if the angle falls in left-hand or right-hand regions
            if 0 < angle <= np.pi:  # Right-hand region (S_r)
                Q_r.append(sheep_position)
            elif -np.pi < angle <= 0:  # Left-hand region (S_l)
                Q_l.append(sheep_position)

        return Q_l, Q_r

    def visible_sheep(self):
        """
        Get all sheep visible to the dog in the current step.
        :return: a list of indices of visible sheep.
        """
        vectors = self.p - self.q
        distances = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        sheep_data = np.array(list(zip(range(len(self.p)), distances, angles)))
        sheep_data = sheep_data[sheep_data[:, 1].argsort()]

        visible_indices = []
        angle_visibility = {}

        for index, distance, angle in sheep_data:
            if angle not in angle_visibility or angle_visibility[angle] > distance:
                angle_visibility[angle] = distance
                if distance <= self.rho_v:
                    visible_indices.append(int(index))
        return self.p[visible_indices]

    def herd_center(self, visible_indices):
        """
        Compute the center of the visible herd.
        """
        if len(visible_indices) > 0:
            visible_positions = self.p[visible_indices]
            center = np.mean(visible_positions, axis=0)
        else:
            center = None
        return center

    def left_most_right_most_sheep(self, visible_sheep, source):
        relative_positions = visible_sheep - source
        angles = np.array([np.arctan2(pos[1], pos[0]) for pos in relative_positions])

        rightmost_sheep = self.p[np.argmin(angles)]  # Position of rightmost sheep
        leftmost_sheep = self.p[np.argmax(angles)]  # Position of leftmost sheep

        # TODO idk if there is a maximum angle or does the dog have 360 deg vision?

        return leftmost_sheep, rightmost_sheep

    def sheep_herd_polygon(self):
        """
        Compute the convex hull of the sheep herd.
        """
        hull = ConvexHull(self.p)
        return self.p[hull.vertices]

    def is_converged(self):
        """
        Check if all sheep have reached the destination (within radius rho_d).
        """
        distances = np.linalg.norm(self.p - self.p_d, axis=1)
        return np.all(distances <= self.rho_d)

def rotation_matrix(theta):
    """
    Returns the 2D rotation matrix for a given angle theta.

    Parameters:
        theta (float): The angle in radians to rotate by.

    Returns:
        np.array: A 2x2 rotation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
    min_x = min(np.min(sim.p[:, 0]), sim.q[0], sim.p_d[0]) - buffer
    max_x = max(np.max(sim.p[:, 0]), sim.q[0], sim.p_d[0]) + buffer
    min_y = min(np.min(sim.p[:, 1]), sim.q[1], sim.p_d[1]) - buffer
    max_y = max(np.max(sim.p[:, 1]), sim.q[1], sim.p_d[1]) + buffer
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Draw the destination circle
    # destination_circle = plt.Circle(sim.p_d, sim.r_d, color='black', fill=False, linewidth=1.5)
    # ax.add_artist(destination_circle)

    # Initialize markers
    sheep_scatter = ax.scatter(sim.p[:, 0], sim.p[:, 1], c='blue', label='Sheep')
    dog_marker, = ax.plot(sim.q[0], sim.q[1], 'ro', label='Dog')

    destination_marker, = ax.plot(sim.p_d[0], sim.p_d[1], 'ro', label='Dog')
    ax.add_artist(destination_marker)

    # Add legend
    ax.legend(loc='upper right')

    step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def update(frame):
        # Run a simulation step
        sim.run_step()

        # Update sheep positions
        sheep_scatter.set_offsets(sim.p)

        # Update dog position
        dog_marker.set_data([sim.q[0]], [sim.q[1]])

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
