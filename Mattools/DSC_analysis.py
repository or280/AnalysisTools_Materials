from typing import List, Dict
import numpy as np


def extract_first_three_columns(file_path: str) -> np.ndarray:
    """
    Extract the first three data columns from a text file and converts time to seconds.

    The expected columns are [Temperature, Time, Heatflow]. This function assumes the
    time column is in minutes and converts it to seconds upon reading.

    Args:
        file_path (str): Path to the input text file

    Rules:
    - Include all rows that are not commented out.
    - A commented line is any non-empty line that starts with '#'.
    - Columns are separated by ';'.
    - Decimal point is '.'.

    Returns:
        np.ndarray: Array of shape (N, 3) with dtype float, where the columns are
                   [Temperature (original unit), Time (in seconds), Heatflow (original unit)].

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
            # Time (column 1) is converted from minutes to seconds by multiplying by 60
            row = [float(parts[0]), float(parts[1]) * 60, float(parts[2])]
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
        raise ValueError(
            f"Expected 2D array with exactly 3 columns [Temperature, Time, Heatflow]; got shape {data.shape}.")
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
        if temp[i - 1] < th_low <= temp[i] and (
                eff_dir[i - 1] > 0 or eff_dir[i - 1] == 0 and eff_dir[min(i, eff_dir.size - 1)] > 0):
            cut_indices.append(i)
            continue
        # Start cooling segment when leaving Tmax by delta downwards
        if temp[i - 1] > th_high >= temp[i] and (
                eff_dir[i - 1] < 0 or eff_dir[i - 1] == 0 and eff_dir[min(i, eff_dir.size - 1)] < 0):
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


def calculate_gradient(segment: np.ndarray, noise_threshold: float = 0.05) -> np.ndarray:
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
                gradient[i + 1:] = np.nan
                break

        # Scan left from the middle
        for i in range(mid_point, 0, -1):
            if abs(gradient[i - 1] - gradient[i]) > noise_threshold:
                gradient[:i] = np.nan
                break
    return gradient


def find_peak_characteristics(
        segment: np.ndarray,
        grad_noise_threshold: float = 0.05,
        baseline_zero_grad_tolerance: float = 0.0002,
        plot_for_troubleshooting: bool = False,
) -> Dict[str, float]:
    """
    Analyzes a DSC segment to find the start, peak, and end temperatures of a transformation,
    and calculates the area of the peak.

    Method:
    1.  The peak temperature is identified as the point of maximum absolute heatflow.
    2.  The gradient of heatflow vs. temperature is calculated.
    3.  The baseline start/end points are found by searching outwards from the peak for the
        first point where the gradient returns to (near) zero.
    4.  The inflection points (start and end of the steepest slope) are found by searching
        for the maximum gradient magnitude between each baseline point and the peak.
    5.  Tangents are constructed at both the inflection points and the baseline points.
    6.  The start and end temperatures are determined by the intersection of the respective
        peak and baseline tangents.
    7.  The area of the peak is calculated by integrating the heatflow curve after
        subtracting a linear baseline constructed between the baseline start and end points.

    Args:
        segment (np.ndarray): A 2D array with shape (N, 3) representing a single
                              heating or cooling segment. Columns: [Temperature, Time, Heatflow].
        grad_noise_threshold (float): Noise threshold for the gradient calculation.
        baseline_zero_grad_tolerance (float): Absolute gradient value below which the
                                              curve is considered to be baseline.
        plot_for_troubleshooting (bool): If True, generates a plot for debugging.

    Returns:
        Dict[str, float]: A dictionary with 'T_start', 'T_peak', 'T_end', and 'Area'.
                          Returns an empty dict if a peak cannot be analyzed.
    """
    if not isinstance(segment, np.ndarray) or segment.ndim != 2 or segment.shape[1] != 3:
        raise ValueError("Input segment must be a 2D numpy array with 3 columns.")
    if segment.shape[0] < 10:  # Need enough points for analysis
        return {}

    temperature = segment[:, 0]
    heatflow = segment[:, 2]

    # 1. Find peak middle (max absolute heatflow)
    peak_idx = np.argmax(np.abs(heatflow))
    t_peak = temperature[peak_idx]

    # 2. Calculate gradient
    gradient = calculate_gradient(segment, noise_threshold=grad_noise_threshold)

    # Handle NaNs from gradient calculation by replacing them with 0
    gradient = np.nan_to_num(gradient, nan=0.0)

    if np.all(gradient == 0):  # Cannot find inflection points if gradient is flat
        return {}

    # 3. Find baseline points by searching outwards from the peak for where the gradient returns to near-zero.
    # To avoid detecting the flat top of a broad peak as the baseline, the search starts a small
    # distance (peak_offset) away from the peak's center.
    peak_offset = 10

    # Search backwards from (peak - offset) for the start of the baseline
    baseline_start_idx = -1
    start_search_before = peak_idx - peak_offset
    # The loop safely handles cases where the start_search_before is out of bounds (e.g. < 0)
    for i in range(start_search_before, -1, -1):
        if abs(gradient[i]) < baseline_zero_grad_tolerance:
            baseline_start_idx = i
            break
    if baseline_start_idx == -1:  # Fallback to the start of the segment
        baseline_start_idx = 0

    # Search forwards from (peak + offset) for the end of the baseline
    baseline_end_idx = -1
    start_search_after = peak_idx + peak_offset
    # The loop safely handles cases where start_search_after is out of bounds
    for i in range(start_search_after, len(gradient)):
        if abs(gradient[i]) < baseline_zero_grad_tolerance:
            baseline_end_idx = i
            break
    if baseline_end_idx == -1:  # Fallback to the end of the segment
        baseline_end_idx = len(gradient) - 1

    # 4. Find inflection points by searching for the maximum gradient between the baseline points and the peak.
    # Define search window for the first inflection point (from baseline start to peak)
    start_search_idx_1 = baseline_start_idx
    end_search_idx_1 = peak_idx

    if start_search_idx_1 >= end_search_idx_1:
        return {}  # Search window is invalid or empty

    grad_before = gradient[start_search_idx_1:end_search_idx_1]
    if grad_before.size == 0:
        return {}

    infl_idx_1 = start_search_idx_1 + np.argmax(np.abs(grad_before))

    # Define search window for the second inflection point (from peak to baseline end)
    start_search_idx_2 = peak_idx
    end_search_idx_2 = baseline_end_idx

    if start_search_idx_2 >= end_search_idx_2:
        return {}  # Search window is invalid or empty

    grad_after = gradient[start_search_idx_2:end_search_idx_2]
    if grad_after.size == 0:
        return {}

    infl_idx_2 = start_search_idx_2 + np.argmax(np.abs(grad_after))

    # Data for baseline tangents
    t_bs, h_bs, g_bs = temperature[baseline_start_idx], heatflow[baseline_start_idx], gradient[baseline_start_idx]
    t_be, h_be, g_be = temperature[baseline_end_idx], heatflow[baseline_end_idx], gradient[baseline_end_idx]

    # 5. Get tangent data for inflection points
    t1, h1, g1 = temperature[infl_idx_1], heatflow[infl_idx_1], gradient[infl_idx_1]
    t2, h2, g2 = temperature[infl_idx_2], heatflow[infl_idx_2], gradient[infl_idx_2]

    # 6. Calculate intersection of peak tangents and baseline tangents
    # Intersection for T_start
    delta_g_start = g1 - g_bs
    if abs(delta_g_start) < 1e-6:
        t_start = t1  # Fallback if tangents are parallel
    else:
        # Intersection of y = g1*(x-t1)+h1 and y = g_bs*(x-t_bs)+h_bs
        t_start = (h_bs - g_bs * t_bs - (h1 - g1 * t1)) / delta_g_start

    # Intersection for T_end
    delta_g_end = g2 - g_be
    if abs(delta_g_end) < 1e-6:
        t_end = t2  # Fallback if tangents are parallel
    else:
        # Intersection of y = g2*(x-t2)+h2 and y = g_be*(x-t_be)+h_be
        t_end = (h_be - g_be * t_be - (h2 - g2 * t2)) / delta_g_end

    # 7. Calculate the area of the peak by integrating after subtracting a linear baseline.
    integration_indices = range(baseline_start_idx, baseline_end_idx + 1)
    peak_area = 0.0
    if len(integration_indices) >= 2:
        temp_for_integration = temperature[integration_indices]
        heatflow_for_integration = heatflow[integration_indices]
        time_for_integration = segment[
            integration_indices, 1
        ]  # Extract time data for the peak

        # Create a linear baseline between the start and end baseline points.
        # np.interp is a convenient way to do linear interpolation.
        baseline_heatflow = np.interp(
            temp_for_integration, [t_bs, t_be], [h_bs, h_be]
        )

        # Subtract baseline from the actual heatflow
        corrected_heatflow = heatflow_for_integration - baseline_heatflow

        # To get area in J/g, we integrate heatflow (W/g) over time (s).
        # This is equivalent to dividing the integral over temperature by the heating rate.
        peak_area = np.trapezoid(corrected_heatflow, time_for_integration)

    # Plotting for troubleshooting
    if plot_for_troubleshooting:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is required for troubleshooting plots. Please install it.")
            return {"T_start": t_start, "T_peak": t_peak, "T_end": t_end, "Area": peak_area}

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the segment data
        ax.plot(temperature, heatflow, label="Segment Data", color="blue", zorder=2)

        # Plot baseline tangents
        bs_tangent_t = np.array([temperature[0], t_start])
        bs_tangent_h = g_bs * (bs_tangent_t - t_bs) + h_bs
        ax.plot(bs_tangent_t, bs_tangent_h, color='grey', linestyle='--', label='Start Baseline Tangent', zorder=1)

        be_tangent_t = np.array([t_end, temperature[-1]])
        be_tangent_h = g_be * (be_tangent_t - t_be) + h_be
        ax.plot(be_tangent_t, be_tangent_h, color='grey', linestyle='-.', label='End Baseline Tangent', zorder=1)

        # Plot peak tangents
        peak_tangent_t1 = np.array([t_start, t1])
        peak_tangent_h1 = g1 * (peak_tangent_t1 - t1) + h1
        ax.plot(peak_tangent_t1, peak_tangent_h1, 'r-', lw=1.5, label='Peak Tangent (Start)', zorder=3)

        peak_tangent_t2 = np.array([t2, t_end])
        peak_tangent_h2 = g2 * (peak_tangent_t2 - t2) + h2
        ax.plot(peak_tangent_t2, peak_tangent_h2, 'g-', lw=1.5, label='Peak Tangent (End)', zorder=3)

        # Mark key points
        ax.plot(t_bs, h_bs, 'kx', markersize=8, mew=2, label='Baseline Start Point')
        ax.plot(t_be, h_be, 'k+', markersize=8, mew=2, label='Baseline End Point')
        ax.plot(t1, h1, 'ro', markersize=6, label='Inflection Point 1', zorder=4)
        ax.plot(t2, h2, 'go', markersize=6, label='Inflection Point 2', zorder=4)

        # Mark characteristic temperatures
        ax.axvline(t_start, color='red', linestyle=':', label=f'T_start = {t_start:.2f}', zorder=1)
        ax.axvline(t_peak, color='purple', linestyle=':', label=f'T_peak = {t_peak:.2f}', zorder=1)
        ax.axvline(t_end, color='green', linestyle=':', label=f'T_end = {t_end:.2f}', zorder=1)

        # Shade the area under the peak for visualization
        if len(integration_indices) >= 2:
            ax.fill_between(
                temp_for_integration,
                baseline_heatflow,
                heatflow_for_integration,
                alpha=0.2,
                color="cyan",
                label=f"Peak Area: {peak_area:.2f}",
            )

        ax.set_title("Peak Characteristics Analysis (Tangential Baseline)")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Heatflow")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()

    return {"T_start": t_start, "T_peak": t_peak, "T_end": t_end, "Area": peak_area}


def analyze_all_segments(
    data_array: np.ndarray, plot_each_segment: bool = False
) -> np.ndarray:
    """
    Analyzes all heating/cooling segments in a DSC data array to find peak characteristics.

    This function first splits the data into segments and then runs peak analysis on each one.

    Args:
        data_array (np.ndarray): The raw (N, 3) DSC data array from extract_first_three_columns.
        plot_each_segment (bool): If True, a troubleshooting plot will be generated for each segment's analysis.

    Returns:
        np.ndarray: A (M, 4) numpy array where M is the number of segments with valid peaks.
                    The columns are ['T_start', 'T_peak', 'T_end', 'Area'].
                    Returns an empty array if no peaks are found.
    """
    segments = split_heating_cooling_segments(data_array)
    if not segments:
        return np.empty((0, 4), dtype=float)

    all_peak_info = []
    for i, segment in enumerate(segments):
        print(f"\n--- Analyzing Segment {i+1} ---")
        peak_info = find_peak_characteristics(
            segment, plot_for_troubleshooting=plot_each_segment
        )
        if peak_info:
            all_peak_info.append(peak_info)
        else:
            print(f"No valid peak found in Segment {i+1}.")

    if not all_peak_info:
        return np.empty((0, 4), dtype=float)

    # Convert list of dicts to a numpy array, ensuring correct column order.
    results_list = [
        [info["T_start"], info["T_peak"], info["T_end"], info["Area"]]
        for info in all_peak_info
    ]
    return np.asarray(results_list, dtype=float)


if __name__ == "__main__":
    import os
    from DSC_plotting import (
        plot_gradient_vs_temperature,
        plot_temperature_vs_heatflow,
    )

    # Define the directory where data files are located.
    # Please update this path to your data folder.
    data_directory = "/Users/oliverreed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - ShapeMemoryAlloyResearch/Useful/Python Scripts/TestFiles"

    filename = input("Enter the file name: ").strip()
    path = os.path.join(data_directory, filename)

    data_array = extract_first_three_columns(path)

    # Plot original data segments first
    if data_array.size > 0:
        segments = split_heating_cooling_segments(data_array)
        if segments:
            print(f"\nPlotting {len(segments)} identified segments.")
            plot_temperature_vs_heatflow(segments, title="DSC Segments")

            # Plot the gradient for the first segment for checking
            '''
            print("\nPlotting gradient for the first segment...")
            first_segment = segments[0]
            gradient = calculate_gradient(first_segment)
            #plot_gradient_vs_temperature(
                first_segment, gradient, title="Gradient for First Segment"
            )
            '''
        else:
            print("No distinct heating/cooling segments found. Plotting raw data.")
            plot_temperature_vs_heatflow(data_array, title="Raw DSC Data")
    else:
        print("No data found in the file.")

    # Analyze all segments and display the results
    if data_array.size > 0:
        print("\nAnalyzing all segments for peak characteristics...")
        # Set plot_each_segment=True to see the analysis for each peak
        all_peaks_data = analyze_all_segments(data_array, plot_each_segment=False)

        if all_peaks_data.size > 0:
            print("\n--- Summary of Peak Characteristics ---")
            print(
                f"{'Segment':<10}{'T_start':<15}{'T_peak':<15}{'T_end':<15}{'Area (J/g)':<15}"
            )
            print("-" * 70)
            for i, row in enumerate(all_peaks_data):
                print(
                    f"{i + 1:<10}{row[0]:<15.2f}{row[1]:<15.2f}{row[2]:<15.2f}{row[3]:<15.2f}"
                )
            print("-" * 70)

            # Save the results to a CSV file
            try:
                base_name, _ = os.path.splitext(filename)
                output_csv_name = f"{base_name}_peak_analysis_results.csv"
                output_csv_path = os.path.join(data_directory, output_csv_name)

                # Add a 'Segment' column to the data for the CSV file
                segment_column = np.arange(1, all_peaks_data.shape[0] + 1).reshape(
                    -1, 1
                )
                data_to_save = np.hstack((segment_column, all_peaks_data))

                header = "Segment,T_start,T_peak,T_end,Area (J/g)"

                np.savetxt(
                    output_csv_path,
                    data_to_save,
                    delimiter=",",
                    header=header,
                    comments="",
                    fmt="%.4f",
                )

                print(f"\nAnalysis results successfully saved to:\n{output_csv_path}")

            except IOError as e:
                print(f"\nError: Could not save results to CSV file. {e}")

        else:
            print("\nNo valid peaks were found in any segment.")