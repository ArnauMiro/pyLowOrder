import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import aerosandbox.tools.pretty_plots as p
from ..utils import raiseWarning


def create_airfoil_optimization_progress_plot(airfoils, rewards, airfoil_name='Airfoil Shape', save_path=None):
    """
    Create a plot showing the evolution of airfoil shapes and their corresponding lift-to-drag ratios.
    
    Args:
        airfoils (List[asb.Airfoil]): List of airfoil objects representing the evolution.
        rewards (List[float]): List of lift-to-drag ratios corresponding to each airfoil.
        airfoil_name (str): Name of the airfoil for the plot title. Default: ``'Airfoil Shape'``.
        save_path (Optional[str]): Path to save the plot. If None, the plot will not be saved. Default: ``None``.
    
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    ax[0].set_title(f"{airfoil_name} Evolution")
    ax[0].set_xlabel("$x/c$")
    ax[0].set_ylabel("$y/c$")

    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Lift-to-Drag Ratio $C_L/C_D$")
    plt.tight_layout()

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        colors=[
            p.adjust_lightness(c, 0.8) for c in
            ["orange", "darkseagreen", "dodgerblue"]
        ]
    )

    colors = cmap(np.linspace(0, 1, len(airfoils)))

    # Plot airfoils
    for i in range(len(airfoils)):
        
        plt.sca(ax[0])
        plt.tight_layout()
        plt.plot(
            airfoils[i].x(),
            airfoils[i].y(),
            "-",
            color=colors[i],
            alpha=0.5,
        )
        plt.axis('equal')

        plt.sca(ax[1])
    
    # Plot rewards
    plt.sca(ax[1])
    p.plot_color_by_value(
        np.arange(len(rewards)),
        np.array(rewards),
        ".-",
        c=np.arange(len(rewards)),
        cmap=cmap,
        clim=(0, len(airfoils)),
        alpha=0.8
    )
    
    # Create a mappable object for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(airfoils)))
    sm.set_array([])
    
    # Add colorbar using the scalar mappable
    cbar = fig.colorbar(sm, ax=ax[1], pad=0.01)
    cbar.set_label('Step')
    
    plt.suptitle("Optimization Progress")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

try: 
    import manim
    class AirfoilEvolutionAnimation(manim.Scene):
        """
        Generate an animation showing the evolution of airfoils and their corresponding lift-to-drag ratios.
        Manim is required to run this animation, and it is recommended to install it with `conda install -c conda-forge manim`.

        Properties:
            - `airfoils`: List of airfoil objects representing the evolution.
            - `rewards`: List of lift-to-drag ratios corresponding to each airfoil.
            - `run_time`: Duration of the animation in seconds. Default: ``5``.
            - `title`: Title of the animation. Default: ``"Airfoil evolution"``.

        Examples:
            To use in a notebook:

            >>> import manim
            >>> from pyLOM.RL import AirfoilEvolutionAnimation

            >>> %%manim -qm -v WARNING AirfoilEvolution
            >>> AirfoilEvolutionAnimation.airfoils = airfoils
            >>> AirfoilEvolutionAnimation.rewards = rewards
            >>> AirfoilEvolutionAnimation.title = "NACA0012 Evolution"

            To use it in a script:

            >>> from pyLOM.RL import AirfoilEvolutionAnimation
            >>> from manim import *
            >>> from main import config
            >>> # config.format = "gif"  # Change output format to GIF
            >>> config.output_file = "airfoil_evolution.mp4"  # Change output file name
            >>> config.quality = "production_quality"  # Set quality to maximum (Full HD)
            >>> animation = AirfoilEvolutionAnimation()
            >>> animation.airfoils = airfoils
            >>> animation.rewards = rewards
            >>> animation.title = "Airfoil Evolution"
            >>> animation.render()
        """
        airfoils = None
        rewards = None
        run_time = 5
        title = "Airfoil evolution"
    
        def get_airfoil_coordinates(self, airfoil, axes):
            """
            Get airfoil coordinates and map them to Axes coordinates.
            Assumes airfoil.coordinates is a (points, 2) numpy array.
            """
            return np.array([axes.c2p(x, y) for x, y in airfoil.coordinates])
        
        def interpolate_airfoils(self, start, end, alpha):
            """Interpolate between two sets of coordinates using alpha."""
            return (1 - alpha) * start + alpha * end
        
        def construct(self):
            import manim
            
            # Configure manim to not use LaTeX
            manim.config.renderer = "cairo"  # Use Cairo renderer instead of LaTeX
            
            # Create axes for airfoil visualization
            axes_airfoil = manim.Axes(
                x_range=[0, 1, 0.2],  # x-axis range with ticks every 0.2
                y_range=[-0.3, 0.3, 0.1],  # y-axis range with ticks every 0.1
                axis_config={
                    "include_numbers": True,
                    "font_size": 50,  # Larger font size to compensate for scaling
                },
                tips=False,  # Remove arrow tips
            )
            
            # Create axis labels using Text instead of Tex
            x_label = manim.Text("x", font_size=55).next_to(axes_airfoil.x_axis, manim.DOWN + manim.RIGHT)
            y_label = manim.Text("y", font_size=55).next_to(axes_airfoil.y_axis, manim.LEFT + manim.UP)
            
            # Create the airfoil shape
            airfoil_shape = manim.VMobject().set_color(manim.BLUE).set_stroke(width=1.5)
            airfoil_shape.set_points_as_corners(self.get_airfoil_coordinates(self.airfoils[0], axes_airfoil))
            self.add(axes_airfoil, x_label, y_label, airfoil_shape)
            
            # Group the airfoil-related objects for transformation
            airfoil_group = manim.VGroup(axes_airfoil, airfoil_shape, x_label, y_label)
            airfoil_group.scale(0.45).move_to(3.5 * manim.LEFT)
            
            # Create axes for the reward plot
            x_step = len(self.rewards) // 10 if len(self.rewards) > 10 else 1
            axes_reward = manim.Axes(
                x_range=[0, len(self.rewards), x_step],  # Iteration range
                y_range=[
                    min(self.rewards) * 0.9, 
                    max(self.rewards) * 1.1, 
                    int((max(self.rewards) * 1.1 - min(self.rewards) * 0.9) / 10)
                ],  # Reward range with padding
                axis_config={
                    "include_numbers": True,
                    "font_size": 50,  # Larger font size to compensate for scaling
                },
                tips=True,
            )
            
            # Create reward plot labels using Text instead of Tex
            iteration_label = manim.Text("Iteration", font_size=55).next_to(axes_reward.x_axis, manim.DOWN)
            reward_label = manim.Text("CL/CD", font_size=55).next_to(axes_reward.y_axis, manim.LEFT + manim.UP)
            
            rewards_group = manim.VGroup(axes_reward, iteration_label, reward_label)
            rewards_group.scale(0.45).move_to(3.5 * manim.RIGHT)
            reward_line = manim.VMobject(color=manim.YELLOW).set_stroke(width=2)
            dot = manim.Dot(color=manim.YELLOW)
            
            def update_reward_line(obj):
                """Updater for the reward plot to grow the line dynamically."""
                total_index = alpha.get_value()
                index = int(total_index)
                t = total_index - index
                # Interpolate rewards for smooth transition
                if index < len(self.rewards) - 1:
                    x_vals = list(range(index + 1))  # Include all completed indices
                    y_vals = self.rewards[:index + 1]
                    # Append interpolated point at fractional step
                    interp_x = index + t
                    interp_y = (1 - t) * self.rewards[index] + t * self.rewards[index + 1]
                    x_vals.append(interp_x)
                    y_vals.append(interp_y)
                    # Update points in the reward line
                    points = [axes_reward.c2p(x, y) for x, y in zip(x_vals, y_vals)]
                    obj.set_points_as_corners(points)
                    dot.move_to(points[-1])
            
            reward_line.add_updater(update_reward_line)
            
            self.add(reward_line, axes_reward, iteration_label, reward_label)
            
            # Add text annotation for airfoil
            airfoil_name = manim.Text(self.title, font_size=42).move_to(3 * manim.UP)
            
            # Create thickness display using Text and DecimalNumber
            thickness_label = manim.Text("Max Thickness = ", font_size=30, color=manim.WHITE)
            thickness_value = manim.DecimalNumber(
                self.airfoils[0].max_thickness(),
                num_decimal_places=4,
                font_size=50,
                color=manim.WHITE
            )
            thickness_display = manim.VGroup(thickness_label, thickness_value).arrange(manim.RIGHT)
            thickness_display.next_to(airfoil_group, manim.DOWN)
            
            # Create a ValueTracker for interpolation
            alpha = manim.ValueTracker(0)
            
            def update_airfoil_shape(obj):
                """Updater to interpolate between airfoil shapes."""
                total_index = alpha.get_value()
                index = int(total_index)  # Determine the starting airfoil
                t = total_index - index  # Fractional part for interpolation
                if index < len(self.airfoils) - 1:
                    start_points = self.get_airfoil_coordinates(self.airfoils[index], axes_airfoil)
                    end_points = self.get_airfoil_coordinates(self.airfoils[index + 1], axes_airfoil)
                    interpolated_points = self.interpolate_airfoils(start_points, end_points, t)
                    obj.set_points_as_corners(interpolated_points)
                    # Update thickness value
                    thickness_value.set_value(self.airfoils[index].max_thickness())
                    
            airfoil_shape.add_updater(update_airfoil_shape)
            
            self.play(
                manim.Write(airfoil_name),
            )
            self.play(
                manim.FadeIn(thickness_display),
                # manim.Write(thickness_display),
            )
            self.add(dot)  # add the dot after writing the title
            
            # Animate the airfoil evolution and reward plot together
            self.play(
                alpha.animate.set_value(len(self.airfoils) - 1), 
                run_time=self.run_time, 
                rate_func=manim.rate_functions.ease_in_circ
            )
            
            # Clean up
            airfoil_shape.remove_updater(update_airfoil_shape)
            reward_line.remove_updater(update_reward_line)
            self.wait()
    
    class WingEvolutionAnimation(manim.ThreeDScene):
        """
        Generate an animation showing the evolution of wings and their corresponding lift-to-drag ratios.
        Manim is required to run this animation, and it is recommended to install it with `conda install -c conda-forge manim`.

        Properties:
            - `wings`: List of wing objects representing the evolution.
            - `rewards`: List of lift-to-drag ratios corresponding to each airplane.
            - `run_time_per_update`: Duration of each update in seconds. Default: ``0.25``.
        
        Examples:
            To use in a notebook:

            >>> import manim
            >>> from pyLOM.RL import WingEvolutionAnimation

            on a separete cell, define the wings and rewards:

            >>> %%manim -qm -v WARNING AirfoilEvolution 
            >>> WingEvolutionAnimation.wings = wings
            >>> WingEvolutionAnimation.rewards = rewards
            >>> WingEvolutionAnimation.run_time_per_update = 0.1
        """
        rewards = None
        wings = None
        run_time_per_update = 0.25

        def get_mesh_from_airplane(self, airplane):
            """Constructs a VGroup of polygons from an airplane's mesh data."""
            import manim
            points, faces = airplane.mesh_body(method='tri')
            # scale the points so that y is normalized to [-1, 1] and the aspect ratio is maintained
            points = np.array(points)
            y_min = points[:, 1].min()
            y_max = points[:, 1].max()
            # Calculate the center of the y-span
            y_center = (y_max + y_min) / 2.0

            # Calculate the scaling factor
            scale_factor = 2.0 / (y_max - y_min)

            # Center the y-axis coordinates around 0
            points[:, 1] = points[:, 1] - y_center
            points *= scale_factor

            mesh_polys = manim.VGroup()
            for face in faces:
                # Get the vertices for this face.
                # the 5 is hardcoded, but it should look fine independently the size of the wing
                triangle_vertices = [points[idx] * 5 for idx in face]
                # Create the polygon.
                triangle = manim.Polygon(
                    *triangle_vertices,
                    fill_color=manim.BLUE,
                    fill_opacity=0.5,
                    stroke_color=manim.WHITE,
                    stroke_width=0.1,
                )
                mesh_polys.add(triangle)
            return mesh_polys
        
        def construct(self):
            import manim

            manim.config.renderer = "cairo"  # Use Cairo renderer instead of LaTeX
            # Set up 3D camera
            self.set_camera_orientation(phi=75 * manim.DEGREES, theta=210 * manim.DEGREES)
            x_step = len(self.rewards) // 10 if len(self.rewards) > 10 else 1
            rewards_axes = manim.Axes(
                x_range=[0, len(self.wings), x_step],
                y_range=[min(self.rewards), max(self.rewards), int((max(self.rewards) * 1.1 - min(self.rewards) * 0.9) / 5)],
                x_length=7,
                y_length=3,
                axis_config={"include_numbers": True},
                tips=False
                
            ).scale(0.85)
            labels_airfoil = rewards_axes.get_axis_labels(x_label=manim.Text("Iteration", font_size=16), y_label=manim.Text("CL/CD", font_size=16))
            # Position the axes at the bottom of the screen
            rewards_group = manim.VGroup(rewards_axes, labels_airfoil)
            rewards_group.to_corner(manim.DOWN + manim.LEFT, buff=0.5)
            
            # Create the rewards plot line
            rewards_points = [rewards_axes.coords_to_point(i, reward) 
                            for i, reward in enumerate(self.rewards)]
            rewards_line = manim.VMobject()
            rewards_line.set_points_smoothly(rewards_points[:1])
            
            # Add the 3D mesh first
            current_mesh = self.get_mesh_from_airplane(self.wings[0]).move_to([0, 0, 1])
            
            self.add(current_mesh)
            
            # Add the 2D elements as fixed in frame
            self.add_fixed_in_frame_mobjects(rewards_group, rewards_line)
            
            self.wait(1)
            
            # Animate through each airplane and update rewards plot
            for i, (airplane, reward) in enumerate(zip(self.wings[1:], self.rewards[1:]), 1):
                new_mesh = self.get_mesh_from_airplane(airplane).move_to([0, 0, 1])
                
                # Create the next segment of the rewards line
                new_line = manim.VMobject()
                new_line.set_points_smoothly(rewards_points[:i+1])
                new_line.set_color(manim.BLUE)  # Match the wing color
                
                # Add the new line as fixed in frame
                self.remove(rewards_line)
                self.add_fixed_in_frame_mobjects(new_line)
                
                # Animate both the mesh transformation and the rewards line update
                self.play(
                    manim.ReplacementTransform(current_mesh, new_mesh),
                    manim.ReplacementTransform(rewards_line, new_line),
                    run_time=self.run_time_per_update,
                )
                
                current_mesh = new_mesh
                rewards_line = new_line
            
            self.wait(1)
            
            # Camera rotation (only affects the 3D part)
            self.move_camera(theta=(2 + 7/6) * manim.PI, run_time=7, rate_func=manim.linear)
            self.wait(1)
    
except:
    class AirfoilEvolutionAnimation:
        """
        Generate an animation showing the evolution of airfoils and their corresponding lift-to-drag ratios.
        Manim is required to run this animation, and it is recommended to install it with `conda install -c conda-forge manim`.

        Properties:
            - `airfoils`: List of airfoil objects representing the evolution.
            - `rewards`: List of lift-to-drag ratios corresponding to each airfoil.
            - `run_time`: Duration of the animation in seconds. Default: ``5``.
            - `title`: Title of the animation. Default: ``"Airfoil evolution"``.

        Examples:
            To use in a notebook:

            >>> import manim
            >>> from pyLOM.RL import AirfoilEvolutionAnimation

            >>> %%manim -qm -v WARNING AirfoilEvolution
            >>> AirfoilEvolutionAnimation.airfoils = airfoils
            >>> AirfoilEvolutionAnimation.rewards = rewards
            >>> AirfoilEvolutionAnimation.title = "NACA0012 Evolution"

            To use it in a script:

            >>> from pyLOM.RL import AirfoilEvolutionAnimation
            >>> from manim import *
            >>> from main import config
            >>> # config.format = "gif"  # Change output format to GIF
            >>> config.output_file = "airfoil_evolution.mp4"  # Change output file name
            >>> config.quality = "production_quality"  # Set quality to maximum (Full HD)
            >>> animation = AirfoilEvolutionAnimation()
            >>> animation.airfoils = airfoils
            >>> animation.rewards = rewards
            >>> animation.title = "Airfoil Evolution"
            >>> animation.render()
        """
        def __init__(self, *args, **kwargs):
            raiseWarning(
                "AirfoilEvolutionAnimation requires manim to be installed."
                "Please, install with: conda install -c conda-forge manim",
                False
            )

    class WingEvolutionAnimation:
        """
        Generate an animation showing the evolution of wings and their corresponding lift-to-drag ratios.
        Manim is required to run this animation, and it is recommended to install it with `conda install -c conda-forge manim`.

        Properties:
            - `wings`: List of wing objects representing the evolution.
            - `rewards`: List of lift-to-drag ratios corresponding to each airplane.
            - `run_time_per_update`: Duration of each update in seconds. Default: ``0.25``.

        Examples:
            To use in a notebook:

            >>> import manim
            >>> from pyLOM.RL import WingEvolutionAnimation

            on a separete cell, define the wings and rewards:

            >>> %%manim -qm -v WARNING AirfoilEvolution 
            >>> WingEvolutionAnimation.wings = wings
            >>> WingEvolutionAnimation.rewards = rewards
            >>> WingEvolutionAnimation.run_time_per_update = 0.1
        """
        def __init__(self, *args, **kwargs):
            raiseWarning(
                "WingEvolutionAnimation requires manim to be installed."
                "Please, install with: conda install -c conda-forge manim",
                False
            )
