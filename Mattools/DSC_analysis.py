from typing import List
import numpy as np

def extract_first_three_columns(file_path: str) -> np.ndarray:
    """
    Extract the first three data columns from a text file.

    Args:
        file_path (str): Path to the input text file

    Rules:
    - Include all rows that are not commented out.
    - A commented line is any non-empty line that starts with '#'.
    - Columns are separated by ';'.
    - Decimal point is '.'.

    Returns:
        np.ndarray: Array of shape (N, 3) with dtype float containing the first
                   three columns of numerical data.

    Examples:
        >>> data = extract_first_three_columns("data.txt")
        >>> print(data.shape)
        (1000, 3)
    """
    # Read all lines
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")

    parsed_data: List[List[float]] = []

    # Parse all non-comment lines
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue  # skip empty lines
        if cleaned_line.startswith("#"):
            # comment line: skip
            continue

        parts = cleaned_line.split(";")
        if len(parts) < 3:
            continue  # not enough columns

        try:
            # Convert only the first three columns to floats
            row = [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            # skip lines that can't be parsed into floats (e.g., residual labels)
            continue

        parsed_data.append(row)

    if not parsed_data:
        return np.empty((0, 3), dtype=float)
    return np.asarray(parsed_data, dtype=float)


def split_heating_cooling_segments(
    data: np.ndarray,
    tol: float = 1e-6,
    delta: float = 3.0,
    min_points: int = 1,
) -> List[np.ndarray]:
    """
    Split DSC data into segments that start when temperature deviates from the global
    minimum or maximum by a specified amount (delta).

    A new heating segment starts when Temperature crosses (Tmin + delta) upward.
    A new cooling segment starts when Temperature crosses (Tmax - delta) downward.

    Parameters:
        data (np.ndarray): 2D array with exactly 3 columns in this order:
                           [Temperature, Time, Heatflow].
        tol (float): Tolerance for determining direction from temperature differences.
        delta (float): Deviation from global min/max that defines the start of a new segment.
        min_points (int): Minimum number of points a segment must have to be kept.

    Returns:
        List[np.ndarray]: A list of segments as numpy arrays, preserving row order.
    """
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected 2D array with exactly 3 columns [Temperature, Time, Heatflow]; got shape {data.shape}.")
    n = data.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [data.copy()]

    # Fixed column order: Temperature (0), Time (1), Heatflow (2)
    temp = data[:, 0]

    # Global extrema and thresholds
    tmin = float(np.min(temp))
    tmax = float(np.max(temp))
    th_low = tmin + float(delta)
    th_high = tmax - float(delta)

    # If thresholds overlap, fall back to a single segment
    if th_low >= th_high:
        return [data.copy()]

    # Differences in temperature between consecutive samples
    dT = np.diff(temp)

    # Determine step-wise direction with tolerance: -1 (cooling), 0 (plateau), 1 (heating)
    step_dir = np.zeros_like(dT, dtype=int)
    step_dir[dT > tol] = 1
    step_dir[dT < -tol] = -1

    # Propagate direction across plateaus (zeros) so they inherit neighboring direction
    eff_dir = step_dir.copy()
    last = 0
    for i in range(eff_dir.size):
        if eff_dir[i] == 0:
            eff_dir[i] = last
        else:
            last = eff_dir[i]
    last = 0
    for i in range(eff_dir.size - 1, -1, -1):
        if eff_dir[i] == 0:
            eff_dir[i] = last
        else:
            last = eff_dir[i]

    # If still all zeros, temperature is effectively constant -> single segment
    if not np.any(eff_dir):
        return [data.copy()]

    # Find indices where we cross the thresholds in the correct direction
    cut_indices: List[int] = []
    for i in range(1, n):
        # Start heating segment when leaving Tmin by delta upwards
        if temp[i - 1] < th_low <= temp[i] and (eff_dir[i - 1] > 0 or eff_dir[i - 1] == 0 and eff_dir[min(i, eff_dir.size - 1)] > 0):
            cut_indices.append(i)
            continue
        # Start cooling segment when leaving Tmax by delta downwards
        if temp[i - 1] > th_high >= temp[i] and (eff_dir[i - 1] < 0 or eff_dir[i - 1] == 0 and eff_dir[min(i, eff_dir.size - 1)] < 0):
            cut_indices.append(i)
            continue

    # Build segments
    segments: List[np.ndarray] = []
    start = 0
    for cut in cut_indices:
        seg = data[start:cut]
        if seg.shape[0] >= min_points:
            segments.append(seg.copy())
        start = cut
    # Append the final segment
    tail = data[start:]
    if tail.shape[0] >= min_points:
        segments.append(tail.copy())

    print(f"Number of segments = {len(segments)}")
    return segments


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


def calculate_gradient(segment: np.ndarray, noise_threshold: float = 1.0) -> np.ndarray:
    """
    Calculates the gradient of Heatflow with respect to Temperature for a given segment.
    This function handles cases where temperature is constant for some data points
    to avoid division by zero errors.

    After calculating the gradient, it filters out noisy regions at the start/end
    of the segment by replacing them with NaNs. The filtering starts from the
    middle and scans outwards.

    Args:
        segment (np.ndarray): A 2D array with shape (N, 3) representing a single
                              heating or cooling segment. The columns must be
                              [Temperature, Time, Heatflow].
        noise_threshold (float, optional): The threshold for point-to-point
                                           change in the gradient to be
                                           considered noise. Defaults to 1.0.

    Returns:
        np.ndarray: A 1D array of shape (N,) containing the gradient
                    d(Heatflow)/d(Temperature), with noisy ends set to NaN.
    """
    if not isinstance(segment, np.ndarray) or segment.ndim != 2 or segment.shape[1] != 3:
        raise ValueError(
            f"Expected a 2D numpy array with 3 columns, but got shape {segment.shape}"
        )

    if segment.shape[0] < 2:
        # Gradient is not well-defined for fewer than 2 points.
        return np.zeros(segment.shape[0], dtype=float)

    temperature = segment[:, 0]
    heatflow = segment[:, 2]

    # Create a mask to select points where the temperature is unique relative to the
    # previous point. This prevents division by zero in np.gradient when temperature
    # is constant over several data points. The first point is always kept.
    mask = np.concatenate(([True], np.diff(temperature) != 0))

    # If there are fewer than 2 points with unique temperatures, we can't calculate a gradient.
    # This handles segments where temperature is constant throughout.
    if np.sum(mask) < 2:
        return np.zeros(segment.shape[0], dtype=float)

    # Calculate the gradient only for the points with unique temperatures.
    # np.gradient uses central differences, avoiding the division-by-zero issue.
    grad_unique = np.gradient(heatflow[mask], temperature[mask])

    # The calculated gradient has the length of the filtered data. We need to map it
    # back to the original array's shape. We use interpolation over the array indices.
    indices = np.arange(segment.shape[0])

    # Interpolate the gradient values across all original indices.
    # - For points that were in the `mask`, np.interp will return their exact gradient value.
    # - For points that were filtered out (constant temperature), it will linearly
    #   interpolate between the gradients of the nearest valid neighbors.
    gradient = np.interp(indices, indices[mask], grad_unique)

    # Noise filtering: find the first large jump from the center outwards and NaN the rest
    if gradient.size > 2 and noise_threshold is not None:
        mid_point = gradient.size // 2

        # Scan right from the middle
        for i in range(mid_point, gradient.size - 1):
            if abs(gradient[i + 1] - gradient[i]) > noise_threshold:
                gradient[i + 1 :] = np.nan
                break

        # Scan left from the middle
        for i in range(mid_point, 0, -1):
            if abs(gradient[i - 1] - gradient[i]) > noise_threshold:
                gradient[:i] = np.nan
                break
    return gradient

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


if __name__ == "__main__":
    path = input("Enter path to the txt file: ").strip()
    data_array = extract_first_three_columns(path)
    segments = split_heating_cooling_segments(data_array)

    # Plot original data segments
    if segments:
        print(f"\nPlotting {len(segments)} identified segments.")
        plot_temperature_vs_heatflow(segments, title="DSC Segments")
    else:
        print("No distinct heating/cooling segments found.")
        # If no segments, but data exists, plot the raw data as a single curve
        if data_array.size > 0:
            print("Plotting raw data.")
            plot_temperature_vs_heatflow(data_array, title="Raw DSC Data")

    # Calculate and plot gradient for the first segment
    if segments:
        # Let's analyze the first segment as an example
        print("\nCalculating and plotting gradient for the first segment...")
        first_segment = segments[0]
        gradient = calculate_gradient(first_segment, noise_threshold=1.0)
        plot_gradient_vs_temperature(
            first_segment,
            gradient,
            title="Gradient of First Segment",
        )
