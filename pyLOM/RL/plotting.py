import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import aerosandbox.tools.pretty_plots as p


def create_airfoil_optimization_progress_plot(airfoils, rewards, airfoil_name='Airfoil Shape', save_path=None):
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

def AirfoilEvolutionAnimation(*args, **kwargs):
    """Factory function that creates AirfoilEvolution instances."""
    try:
        from manim import Scene
    except ImportError as e:
        raise ImportError(
            "AirfoilEvolution requires manim to be installed."
            "Please, install with: conda install -c conda-forge manim"
        ) from e
    
    class _AirfoilEvolutionAnimation(Scene):
        airfoils = AirfoilEvolutionAnimation.airfoils
        rewards = AirfoilEvolutionAnimation.rewards
        run_time = AirfoilEvolutionAnimation.run_time if hasattr(AirfoilEvolutionAnimation, "run_time") else 5
        title = AirfoilEvolutionAnimation.title if hasattr(AirfoilEvolutionAnimation, "title") else "Airfoil evolution"
        
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
    
    return _AirfoilEvolutionAnimation(*args, **kwargs)