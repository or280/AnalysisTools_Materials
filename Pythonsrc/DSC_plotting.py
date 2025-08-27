import numpy as np
from typing import List


def plot_temperature_vs_heatflow(
    segments_or_data: np.ndarray | List[np.ndarray],
    show: bool = True,
    save_path: str | None = None,
    title: str = "Heatflow vs Temperature",
    xlabel: str = "Temperature",
    ylabel: str = "Heatflow",
) -> None:
    """
    Plot Heatflow (y) vs Temperature (x).
    - If given a single Nx3 array, plot it as a single black curve.
    - If given a list of segments (each Nx3), plot each segment:
      cooling segments in blues, heating segments in reds, darker for later cycles.
    """
    try:
        import matplotlib.pyplot as plt  # local import to avoid hard dependency when plotting isn't used
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Please install it to use plot_temperature_vs_heatflow."
        ) from e

    # Case 1: single array -> plot once in black
    if isinstance(segments_or_data, np.ndarray):
        data = segments_or_data
        if data.size == 0:
            raise ValueError("No data to plot: the input array is empty.")
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"Expected data with at least 3 columns, got shape {data.shape}.")

        x = data[:, 0]
        y = data[:, 2]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, lw=1.2, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    # Case 2: list of segments -> color by direction (start vs end temperature)
    if isinstance(segments_or_data, list):
        segments = segments_or_data
        if len(segments) == 0:
            raise ValueError("No segments to plot: the input list is empty.")

        def seg_dir(seg: np.ndarray) -> int:
            if seg.shape[0] < 2:
                return 0
            d = seg[-1, 0] - seg[0, 0]
            if d > 0:
                return 1  # heating
            if d < 0:
                return -1  # cooling
            return 0  # flat/unknown

        directions = [seg_dir(seg) for seg in segments]
        n_heat = sum(1 for d in directions if d == 1)
        n_cool = sum(1 for d in directions if d == -1)

        fig, ax = plt.subplots(figsize=(8, 5))
        reds = plt.cm.Reds
        blues = plt.cm.Blues

        def shade(cmap, idx: int, total: int):
            if total <= 1:
                frac = 0.7
            else:
                frac = 0.3 + 0.6 * (idx / (total - 1))  # darker for later cycles
            return cmap(frac)

        heat_i = 0
        cool_i = 0

        for seg, d in zip(segments, directions):
            if seg.ndim != 2 or seg.shape[1] < 3:
                raise ValueError(f"Each segment must have at least 3 columns, got shape {seg.shape}.")
            x = seg[:, 0]
            y = seg[:, 2]
            if d == 1:
                color = shade(reds, heat_i, n_heat)
                heat_i += 1
            elif d == -1:
                color = shade(blues, cool_i, n_cool)
                cool_i += 1
            else:
                color = "0.5"  # grey for unknown/flat
            ax.plot(x, y, lw=1.2, color=color)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    raise TypeError("segments_or_data must be either a numpy array or a list of numpy arrays (segments).")


def plot_gradient_vs_temperature(
    segment: np.ndarray,
    gradient: np.ndarray,
    show: bool = True,
    save_path: str | None = None,
    title: str = "Heatflow Gradient vs Temperature",
    xlabel: str = "Temperature",
    ylabel: str = "d(Heatflow)/d(Temperature)",
) -> None:
    """
    Plots the gradient of heatflow against temperature for a single segment.
    NaN values in the gradient are automatically handled by matplotlib, creating
    gaps in the plot.

    Args:
        segment (np.ndarray): The segment data, a 2D array of shape (N, 3) with
                              columns [Temperature, Time, Heatflow].
        gradient (np.ndarray): A 1D array of shape (N,) containing the gradient
                               values corresponding to each point in the segment.
        show (bool): If True, display the plot.
        save_path (str | None): If provided, save the plot to this path.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Please install it to use this function."
        ) from e

    if not isinstance(segment, np.ndarray) or segment.ndim != 2 or segment.shape[1] != 3:
        raise ValueError(f"Expected segment to be a 2D numpy array with 3 columns, but got shape {segment.shape}")
    if not isinstance(gradient, np.ndarray) or gradient.ndim != 1:
        raise ValueError(f"Expected gradient to be a 1D numpy array, but got shape {gradient.shape}")
    if segment.shape[0] != gradient.shape[0]:
        raise ValueError(f"Segment and gradient must have the same number of points, but got {segment.shape[0]} and {gradient.shape[0]}")

    if segment.size == 0:
        raise ValueError("No data to plot: the input segment is empty.")

    temperature = segment[:, 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temperature, gradient, lw=1.2, color="purple")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.01)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)