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