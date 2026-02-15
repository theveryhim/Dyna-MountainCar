import numpy as np
import matplotlib.pyplot as plt
import pygame

class CustomMountainCar:
    """
    A customized Mountain Car environment with continuous state space,
    optional extra terrain curvature, learned perturbation dynamics,
    and support for model-based prediction and interactive control.
    """

    def __init__(self, extra_curve=False, extra_const=1.0):
        """
        Initialize the Custom Mountain Car environment.

        Parameters
        ----------
        extra_curve : bool, optional
            If True, adds an additional nonlinear terrain component
            to increase environment complexity.
        extra_const : float, optional
            Scaling factor for the extra terrain perturbation.
        """
         
        # Physical parameters
        self.g = 9.8
        self.dt = 1.0 / 30.0
        self.max_force = 4.0

        # State
        self.x = 0.0
        self.v = 0.0

        # Terrain
        self.extra_curve = extra_curve

        # Limits (adjust as needed)
        self.x_min = -7.0
        self.x_max = 10.0
        self.v_min = -15.0
        self.v_max = 15.0

        self.max_height = 20#max(self.h(x) for x in np.linspace(self.x_min, self.x_max, 1000))
        self.goal_x = 10
        self.extra_const = extra_const
        self.max_steps = 1000

        self.N_X_BINS = 100
        self.f_perturb = np.zeros(self.N_X_BINS)
        self.f_counts  = np.zeros(self.N_X_BINS)  # for averaging

    # -----------------------------
    # Terrain height
    # -----------------------------
    def h(self, x):
        """
        Compute terrain height at position x.

        Parameters
        ----------
        x : float
            Horizontal position.

        Returns
        -------
        float
            Terrain height h(x).
        """
        h = 5 * np.sin(np.pi * x / 10 - np.pi / 2) + 5
        if self.extra_curve:
            if x > 0:
                h += self.extra_const *( np.sin(4 * np.pi * x / 10 - np.pi / 2) + x/10)
            else:
                h -= self.extra_const
        return h

    def x_to_bin(self, x):
        """
        Map a continuous position x to a discrete bin index.

        Parameters
        ----------
        x : float
            Continuous x-position.

        Returns
        -------
        int
            Discrete bin index in [0, N_X_BINS-1].
        """
        bin_idx = int(
            (x - self.x_min) / (self.x_max - self.x_min) * self.N_X_BINS
        )
        return np.clip(bin_idx, 0, self.N_X_BINS - 1)
    
    def bin_to_x(self, bin_idx):
        """
        Convert a discrete x-bin index to the center x-position.

        Parameters
        ----------
        bin_idx : int
            Discrete bin index.

        Returns
        -------
        float
            Continuous x-position corresponding to the bin center.
        """
        bin_width = (self.x_max - self.x_min) / self.N_X_BINS
        return self.x_min + (bin_idx + 0.5) * bin_width

    
    def update_f_perturb(self, x, v, v_next, action):
        """
        Update the learned force perturbation model using an observed transition.

        This estimates unknown dynamics (model error) by subtracting known
        physics from observed acceleration.

        Parameters
        ----------
        x : float
            Current position.
        v : float
            Current velocity.
        v_next : float
            Next velocity after the step.
        action : float
            Applied action in [-1, 1].
        """
        F = np.clip(action, -1.0, 1.0) * self.max_force

        slope = self.dh_dx(x)
        sin_theta = slope / np.sqrt(1.0 + slope ** 2)

        dv_real = (v_next - v) / self.dt
        dv_known = F - self.g * sin_theta

        f_hat = dv_real - dv_known   # <-- inferred perturbation

        idx = self.x_to_bin(x)

        # incremental average (stable)
        self.f_counts[idx] += 1
        #alpha = 1.0 / self.f_counts[idx]
        self.f_perturb[idx] =  f_hat

    def predict(self, state, action_idx):
        """
        Predict the next state and reward using the learned dynamics model.

        This function is intended for model-based RL algorithms
        such as Dyna-Q.

        Parameters
        ----------
        state : tuple (x, v)
            Current continuous state.
        action_idx : int
            Discrete action index (0,1,2) mapped to {-1,0,1}.

        Returns
        -------
        tuple or None
            ((x_next, v_next), reward, done) if the model is available,
            otherwise None.
        """
        x, v = state
        action = action_idx - 1.0
        F = np.clip(action, -1.0, 1.0) * self.max_force

        slope = self.dh_dx(x)
        sin_theta = slope / np.sqrt(1.0 + slope ** 2)

        # velocity update
        idx = self.x_to_bin(self.x)

        # if the state seen in the model use it
        if self.f_counts[idx] > 0:
            f_extra = self.f_perturb[idx]
        else:
            return None
        v += (F - self.g * sin_theta + f_extra) * self.dt
        v = np.clip(v, self.v_min, self.v_max)

        # position update
        x += v * self.dt * 1 / np.sqrt(1.0 + slope ** 2)
        x = np.clip(x, self.x_min, self.x_max)
        if(np.abs(x - self.x_min)<0.01):
            v = 0.2

        # example termination
        done = x >= self.x_max - 0.2

        # example reward
        reward = -1.0

        
        return (x, v), reward, done
    
    # -----------------------------
    # Terrain slope dh/dx
    # -----------------------------
    def dh_dx(self, x):
        """
        Compute the derivative of terrain height dh/dx at position x.

        Parameters
        ----------
        x : float
            Horizontal position.

        Returns
        -------
        float
            Terrain slope at position x.
        """
        slope = (np.pi / 2) * np.sin(np.pi * x / 10)
        #if self.extra_curve:
        #    slope += (3 * np.pi / 5) * np.sin(2 * np.pi * x / 10)
        if self.extra_curve:
            if x > 0:
                slope += self.extra_const *( 4 * np.pi /10 * np.cos(4 * np.pi * x / 10 - np.pi / 2) + 1/10)
        return slope

    # -----------------------------
    # Step
    # -----------------------------
    def step(self, action):
        """
        Advance the environment by one time step.

        Parameters
        ----------
        action : float
            Continuous action in [-1, 0, +1] representing engine force.

        Returns
        -------
        state : np.ndarray
            Next state [x, v].
        reward : float
            Reward for the transition.
        done : bool
            Whether the episode has terminated.
        """
        # action ∈ [-1, 0, +1] (typical MountainCar)
        F = np.clip(action, -1.0, 1.0) * self.max_force

        slope = self.dh_dx(self.x)
        sin_theta = slope / np.sqrt(1.0 + slope ** 2)

        # velocity update
        self.v += (F - self.g * sin_theta) * self.dt
        self.v = np.clip(self.v, self.v_min, self.v_max)

        # position update
        self.x += self.v * self.dt * 1 / np.sqrt(1.0 + slope ** 2)
        self.x = np.clip(self.x, self.x_min, self.x_max)
        if(np.abs(self.x - self.x_min)<0.01):
            self.v = 0.2

        # example termination
        done = self.x >= self.x_max - 0.2
        if self.idx_step > self.max_steps:
            done = True

        # example reward
        reward = -1.0
        self.idx_step += 1
        return np.array([self.x, self.v]), reward, done

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns
        -------
        np.ndarray
            Initial state [x, v].
        """
        self.idx_step = 0
        self.x = np.random.uniform(-1.0, 1.0)
        self.v = 0.0
        return np.array([self.x, self.v])
    
    def sample(self):
        """
        Sample a previously observed state-action pair
        for model-based planning.

        Returns
        -------
        tuple or None
            ((x, v), action) if available, otherwise None.
        """
        valid_indices = self.f_counts > 0
        if len(valid_indices) > 0:
            idx = np.random.choice(valid_indices)
            x = self.bin_to_x(idx)
            v = np.random.rand()*(self.v_max-self.v_min) + self.v_min
            a = np.random.randint(0,3)-1
            return (x,v), a
        else:
            return None

    def plot(self, x_range=(-7, 10), num_points=1000):
        """
        Plot the mountain terrain and the current car position.

        Parameters
        ----------
        x_range : tuple, optional
            Range of x values to plot.
        num_points : int, optional
            Number of points used to draw the terrain.
        """
        xs = np.linspace(x_range[0], x_range[1], num_points)
        ys = np.array([self.h(x) for x in xs])

        # Car position
        car_x = self.x
        car_y = self.h(car_x)

        plt.figure(figsize=(10, 4))

        # Plot mountain curve
        plt.plot(xs, ys, label="Mountain Curve", color="black")

        # Plot car
        plt.scatter(car_x, car_y, color="red", s=80, zorder=3, label="Car")

        # Labels
        plt.xlabel("x (position)")
        plt.ylabel("h(x) (height)")
        plt.title(
            "Custom Mountain Car Terrain"
            + (" + Extra Curve" if self.extra_curve else "")
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def play_with_pygame(self, width=1000, height=400):
        """
        Run an interactive Mountain Car simulation using Pygame.

        Controls
        --------
        A : accelerate left
        D : accelerate right
        Q / ESC : quit

        Parameters
        ----------
        width : int
            Window width in pixels.
        height : int
            Window height in pixels.
        """
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Mountain Car – Pygame")
        clock = pygame.time.Clock()

        font = pygame.font.SysFont(None, 24)

        # World → screen mapping
        def world_to_screen(x, y):
            px = int((x - self.x_min) / (self.x_max - self.x_min) * width)
            py = int(height - y / self.max_height * height) - 100
            return px, py

        # Precompute terrain polyline
        xs = np.linspace(self.x_min, self.x_max, 1000)
        ys = np.array([self.h(x) for x in xs])
        terrain_points = [world_to_screen(x, y) for x, y in zip(xs, ys)]

        self.reset()
        action = 0.0
        running = True

        while running:
            clock.tick(30)  # ~30 FPS

            # --------------------
            # Events
            # --------------------
            

            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_d:
                            action = 1.0
                        elif event.key == pygame.K_a:
                            action = -1.0
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

                    elif event.type == pygame.KEYUP:
                        keys = pygame.key.get_pressed()

                        if event.key == pygame.K_d:
                            if keys[pygame.K_a]:
                                action = -1.0
                            else:
                                action = 0.0

                        elif event.key == pygame.K_a:
                            if keys[pygame.K_d]:
                                action = 1.0
                            else:
                                action = 0.0

            # --------------------
            # Step physics
            # --------------------
            state, reward, done = self.step(action)
            self.x = state[0]
            # Reset on success
            if done:
                self.reset()

            # --------------------
            # Draw
            # --------------------
            screen.fill((240, 240, 240))

            # Terrain
            pygame.draw.lines(screen, (0, 0, 0), False, terrain_points, 2)

            # Car
            car_x = self.x
            car_y = self.h(car_x)
            cx, cy = world_to_screen(car_x, car_y)
            pygame.draw.circle(screen, (200, 30, 30), (cx, cy - 6), 8)

            # Goal
            gx, gy = world_to_screen(self.goal_x, self.h(self.goal_x))
            pygame.draw.circle(screen, (0, 180, 0), (gx, gy - 6), 10, 2)

            # HUD
            text = font.render(
                f"x={self.x:.2f}  v={self.v:.2f}  action={action:+.1f}",
                True,
                (0, 0, 0)
            )
            screen.blit(text, (10, 10))

            pygame.display.flip()

        pygame.quit()


#env = CustomMountainCar(extra_curve=True, extra_const=3.0)
#env.reset()
#env.play_with_pygame()
#env.plot()