"""Data Analyzer GUI for time-series signals acquired from CSV files.

Features:
- Load time-series data (time + multiple channels) from CSV (tab- or comma-separated).
- Interactively select channels and plot them vs. time (with optional normalization,
  detrending, and calibration).
- Interactively select a time window by clicking on the plot.
- Compute and display:
  - Power Spectral Density (PSD) using Welch's method.
  - Probability density functions (PDFs) via histograms (linear and log scale).
  - Combined PDFs across multiple loaded files for comparison (e.g. "free" vs "cell").
- Save:
  - The selected time window of the chosen signals to CSV.
  - The most recently computed PSD to CSV.
- Provide calibration handling for specific analog input channels (piezo, QPD, laser),
  with a configurable conversion factor (CF).
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
#from numba import njit
from scipy.signal import welch, detrend, savgol_filter # For PSD calculation and detrending
from scipy.optimize import curve_fit
import matplotlib.patches as patches # Import patches for Rectangle type checking
import os # Import os module for path manipulation

# --- Data Processing Functions ---
def compute_first_passage_times(t:np.ndarray, x:np.ndarray)->np.ndarray:
    """Compute a boolean mask indicating first-passage events in a time series.
    
    The signal is smoothed with a running average, then detrended. Zero-crossings
    where the detrended signal goes from <= 0 to > 0 and exceeds a small positive
    threshold are marked as first-passage events.
    """
    # This function seems to be for a specific analysis, not directly PSD.
    # We will keep it for now but it might not be directly used for PSD.
    x_mean = np.mean(x)
    delta_t_mean = t[0] + 0.5 #s
    N = np.cumsum(t < delta_t_mean)
    N = N[-1]
    kernel =  np.ones((N,))/N
    x_mean = np.convolve(x,kernel)[N-1:]

    dx = x - x_mean

    a1 = dx[1:] > 0.0
    a2 = dx[:-1] <= 0.0
    i_pass_zero =  a1 & a2
    i_pass_zero = np.insert(i_pass_zero, 0, 0)


    v_idx = (dx) > 0.0015
    idx = v_idx &  i_pass_zero
    #idx = np.insert(idx, 0, 0)

    return idx

def plot_time_range_in_ax(ax, t, piezo1, t_idx):
    """Plot a histogram of first-passage time intervals for a given time window.
    
    The function restricts t and signal values to the selected indices, detects
    first-passage events using compute_first_passage_times(), computes time
    differences between successive events, and plots their histogram.
    """
    t = t[t_idx]
    piezo1 = piezo1[t[t_idx]]

    
    idx = compute_first_passage_times(t,piezo1)
    (fidx, ) = np.where(idx)
    start_idx = 0
    end_idx = -1
    rng1 = np.arange(fidx[start_idx], fidx[end_idx])
    rng2 = fidx[start_idx:end_idx] - fidx[start_idx]
    t2 = t[rng1]
    piezo12 = piezo1[rng1]

    ax.hist(np.diff(t[fidx]), bins=200)

def plot_details_in_ax(ax, t, piezo1, t_idx):
    """Plot the signal segment around first-passage events for visual inspection.
    
    The function restricts the time and signal arrays to the selected indices,
    detects first-passage events, and highlights the event samples on top of
    the raw signal trace.
    """
    t = t[t_idx]
    piezo1 = piezo1[t_idx]

    
    idx = compute_first_passage_times(t,piezo1)
    (fidx, ) = np.where(idx)
    start_idx = 0
    end_idx = -1
    rng1 = np.arange(fidx[start_idx], fidx[end_idx])
    rng2 = fidx[start_idx:end_idx] - fidx[start_idx]
    t2 = t[rng1]
    piezo12 = piezo1[rng1]

    ax.plot(t2, piezo12)
    ax.plot(t2[rng2], piezo12[rng2], linestyle='', marker='o')

def make_axes_xlim_equals(ax):
    """Synchronize the x-axis limits across a list of Matplotlib Axes.
    
    The function finds the global minimum and maximum x-limits and applies them
    to each axis in the provided list.
    """
    if len(ax) == 0: return
    try:
        (mn, mx) = ax[0].get_xlim()
        for a in ax:
            (tmn, tmx) = a.get_xlim()
            mn = min(mn, tmn)
            mx = max(mx, tmx)

        for a in ax:
            a.set_xlim(mn, mx)
    except Exception as e:
        print(f"Error in make_axes_xlim_equals: {e}")


# --- GUI Application Class ---

class DataAnalyzerApp:
    """Main GUI application for loading, visualizing, and analyzing time-series data.

    The class builds the Tkinter interface, manages data loading and preprocessing,
    enables interactive time-window selection, runs spectral and histogram-based
    analyses, and provides options to export selected data and PSD results.
    """
    def __init__(self, root):
        """Initialize the GUI application and internal state.
        
        Creates Tk variables, default parameters, and calls helper methods to
        construct the UI and wire up event handlers.
        """
        self.root = root
        self.root.title("Data Analyzer")
        self.root.geometry("1200x800")

        self.data = None
        self.time_column = None
        self.signal_columns = []
        self.selected_signals = []
        self.file_path = tk.StringVar()
        self.selected_time_range = None # To store (start_time, end_time)
        self.start_line = None # To store the vertical line for selection start
        self.end_line = None   # To store the vertical line for selection end
        self.highlight_patches = [] # To store axvspan patches for easy removal

        # Variables to store PSD data for saving
        self.last_computed_freqs = None
        self.last_computed_psd = None
        self.last_computed_signal_name = None

        # Variable to remember the last opened directory
        self.last_opened_directory = "." 

        # Preprocessing variables
        self.normalize_var = tk.BooleanVar(value=False)
        self.detrend_var = tk.BooleanVar(value=False) # New detrend variable
        self.calibrate_var = tk.BooleanVar(value=False) #New variable to calibrate or not the selected signal
        self.cf_var = tk.DoubleVar(value=0.0)
        self.reference_signal_var = tk.StringVar()

        # === USER SETTINGS ===
        self.SAMPLING_FREQ_HZ = 10000.0   # Hz
        self.NPERSEG = 4096               # Welch segment length
        self.HIST_BINS = 100              # Histogram bins
        #self.BIN_RANGE = (-350, 350)      # Histogram absciss limits in nanometers
        self.BIN_RANGE = (-100, 100)      # Histogram absciss limits in nanometers
        self.K_BOLTZMANN = 4.1            # Boltzmann constant in pN·nm
        self.CF= 1.6;                     # QPD calibration factor correction
        self.cf_var.set(self.CF)
        self.CALIBRATION_FACTORS = [2.0,                    # AI1 - piezo - 2.0 um / V
                                    1.0,                    # AI2 - QPD sum - V
                                    1000.0 * self.CF,       # AI3 - QPD XDIFF - nm / V
                                    1000.0 * self.CF,       # AI4 - QPD XDIFF - nm / V
                                    2.0,                    # AI5 - piezo - 2.0 um / V
                                    1.0,                    # AI6 - LASER correction voltage - V
                                    1.0                     # AI7 - LASER actual voltage - V
                                    ]
        self.CALIBRATION_UNITS = ["um", "AU", "nm", "nm", "um", "V", "V"]
        self.SIGNAL_TITLES = ["Piezo displacement", "Z displacemnt", "X displacement", "Y displacement", "Piezo displacement", "Laser correction", "Laser level"]

        self.data_list = list() #list with data (e.g. histogram data) used to plot combined data from multiple files in same figure for comparison
        # ======================

        self.create_widgets()
        self.setup_event_handlers()

    def create_widgets(self):
        """Construct and lay out all GUI widgets.
        
        Builds the left control pane (file selection, signal list, preprocessing
        controls and analysis buttons) and the right pane with the embedded
        Matplotlib figure and toolbar.
        """
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Panedwindow for left (controls) and right (plot)
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Left Pane (Controls) ---
        control_frame = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(control_frame, weight=1)

        # File selection
        file_frame = ttk.LabelFrame(control_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)

        # Signal selection
        signal_frame = ttk.LabelFrame(control_frame, text="Select Signals", padding="10")
        signal_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.signal_listbox = tk.Listbox(signal_frame, selectmode=tk.MULTIPLE, exportselection=False)
        self.signal_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.signal_listbox.bind('<<ListboxSelect>>', self.on_signal_select)

        # Add a scrollbar for the listbox
        signal_scrollbar = ttk.Scrollbar(signal_frame, orient=tk.VERTICAL, command=self.signal_listbox.yview)
        signal_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.signal_listbox.config(yscrollcommand=signal_scrollbar.set)

        # --- Plotting Controls Frame (Container for Plotting and Preprocessing) ---
        plot_controls_container_frame = ttk.LabelFrame(control_frame, text="Controls", padding="10")
        plot_controls_container_frame.pack(fill=tk.X, pady=5)

        # Plotting Controls Sub-frame
        plotting_sub_frame = ttk.LabelFrame(plot_controls_container_frame, text="Plotting", padding="10")
        plotting_sub_frame.pack(fill=tk.X, pady=2)

        ttk.Button(plotting_sub_frame, text="Plot Selected Signals", command=self.plot_selected_signals).pack(pady=2)
        ttk.Button(plotting_sub_frame, text="Compute PSD", command=self.compute_psd).pack(pady=2)
        ttk.Button(plotting_sub_frame, text="Save Selected Data", command=self.save_selected_data).pack(pady=2)
        ttk.Button(plotting_sub_frame, text="Save PSD Data", command=self.save_psd_data).pack(pady=2)
        ttk.Button(plotting_sub_frame, text="Histogram PDF", command=self.compute_histogram_and_plots).pack(pady=2)
        ttk.Button(plotting_sub_frame, text="Plot combined pdf", command=self.plot_combined_pdf).pack(pady=2)
        

        # Preprocessing Controls Sub-frame
        preprocessing_frame = ttk.LabelFrame(plot_controls_container_frame, text="Preprocessing", padding="10")
        preprocessing_frame.pack(fill=tk.X, pady=2)

        # put a grid frame in the preprocessing_frame to accomodate more controls
        preprocessing_checkbuttons_frame = ttk.Frame(preprocessing_frame, padding="10")
        preprocessing_checkbuttons_frame.pack(fill=tk.X, pady=2)

        ttk.Checkbutton(preprocessing_checkbuttons_frame, text="Normalize Signals", variable=self.normalize_var, command=self.on_normalize_toggle).grid(row=0, column=0, pady=2, sticky='W')
        ttk.Checkbutton(preprocessing_checkbuttons_frame, text="Detrend Signals", variable=self.detrend_var, command=self.on_detrend_toggle).grid(row=1, column=0,  pady=2, sticky='W')
        ttk.Checkbutton(preprocessing_checkbuttons_frame, text="Calibrate Signals", variable=self.calibrate_var, command=self.on_calibrate_toggle).grid(row=2, column=0,  pady=2, sticky='W')

        #let the user modify the CF used in computing CALIBRATION_FACTORS
        cf_frame = ttk.Frame(preprocessing_checkbuttons_frame)
        cf_frame.grid(row=2, column=1,  pady=2, sticky='W')

        ttk.Label(cf_frame, text="CF").pack(side='left', pady=2)
        self.cf_entry = ttk.Entry(cf_frame, textvariable=self.cf_var)
        self.cf_entry.pack(side='left', pady=2)
        self.cf_entry.bind('<Return>', self.update_conversion_factors_from_cf)
        self.cf_entry.bind('<FocusOut>', self.update_conversion_factors_from_cf, add='+')


        #ttk.Checkbutton(preprocessing_frame, text="Normalize Signals", variable=self.normalize_var, command=self.on_normalize_toggle).pack(anchor=tk.W, pady=2)
        #ttk.Checkbutton(preprocessing_frame, text="Detrend Signals", variable=self.detrend_var, command=self.on_detrend_toggle).pack(anchor=tk.W, pady=2) # New detrend checkbox
        #ttk.Checkbutton(preprocessing_frame, text="Calibrate Signals", variable=self.calibrate_var, command=self.on_calibrate_toggle).pack(anchor=tk.W, pady=2)

        ref_signal_label = ttk.Label(preprocessing_frame, text="Reference Signal:")
        ref_signal_label.pack(anchor=tk.W, pady=2)
        self.reference_signal_combobox = ttk.Combobox(preprocessing_frame, textvariable=self.reference_signal_var, state="readonly")
        self.reference_signal_combobox.pack(fill=tk.X, pady=2)
        self.reference_signal_combobox.bind("<<ComboboxSelected>>", self.on_reference_signal_select)


        # --- Right Pane (Plotting Area) ---
        plot_frame = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(plot_frame, weight=3)

        # Matplotlib figure and canvas
        # We will use a single axes for now to simplify time selection, can expand later
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6)) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Initial plot setup
        self.ax.set_title("Select Signals and Time Range")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.fig.tight_layout()
        self.canvas.draw()

    def setup_event_handlers(self):
        """Attach Matplotlib event callbacks for interactive time selection.
        
        Left-click defines the start and end of the time window; right-click
        clears the current selection.
        """
        # Bind mouse events for time selection
        # FIX: Use self.canvas instead of self.canvas_widget for mpl_connect
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.canvas.mpl_connect('button_release_event', self.on_plot_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_plot_motion)

    def browse_file(self):
        """Open a file dialog to select a CSV file and load it.
        
        Updates the file path entry, remembers the last opened directory, and
        calls load_data() to parse the chosen file.
        """
        f_path = filedialog.askopenfilename(
            initialdir=self.last_opened_directory, # Use last opened directory
            title="Select CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if f_path:
            self.file_path.set(f_path)
            self.last_opened_directory = os.path.dirname(f_path) # Update last opened directory
            self.load_data(f_path)

    def load_data(self, f_path):
        """Load time-series data from the given CSV file.
        
        Attempts both tab- and comma-separated formats, detects the time column,
        computes sampling frequency, populates the signal listbox and reference
        signal combobox, and triggers the initial plot.
        """
        try:
            # Explicitly reset time selection markers and patches before loading new data
            self.reset_time_selection() 

            # Try reading with tab delimiter first, then comma
            try:
                self.data = pd.read_csv(f_path, delimiter='\t')
            except Exception:
                self.data = pd.read_csv(f_path, delimiter=',')

            # Identify time and signal columns
            potential_time_cols = [col for col in self.data.columns if 'time' in col.lower()]
            if potential_time_cols:
                self.time_column = potential_time_cols[0]
                # Compute sampling frequency as mean(diff(time_values))
                self.SAMPLING_FREQ_HZ = np.mean(np.diff(self.data[self.time_column].values))
            else:
                # Fallback to the first column if no clear time column is found
                self.time_column = self.data.columns[0]
                print(f"Warning: No clear time column found. Using '{self.time_column}' as time column.")

            self.signal_columns = [col for col in self.data.columns if col != self.time_column]

            # Update signal selection listbox
            self.signal_listbox.delete(0, tk.END)
            for i, col in enumerate(self.signal_columns):
                self.signal_listbox.insert(tk.END, col)
                # Pre-select ONLY the first channel by default
                if i == 0: # Changed from i < 7 to i == 0
                    self.signal_listbox.selection_set(i)
            
            # Update selected_signals based on initial selection
            # Directly update self.selected_signals and call plot_selected_signals
            if self.signal_columns: # Ensure there are columns to select
                self.selected_signals = [self.signal_columns[0]] # Select only the first signal
            else:
                self.selected_signals = []

            # Populate reference signal combobox
            self.reference_signal_combobox['values'] = self.signal_columns
            if 'AI2' in self.signal_columns:
                self.reference_signal_var.set('AI2')
            elif self.signal_columns:
                self.reference_signal_var.set(self.signal_columns[0]) # Default to first signal if AI2 not found
            else:
                self.reference_signal_var.set('')
            
            # Call plot_selected_signals to reflect this initial selection
            self.plot_selected_signals() 

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.data = None
            self.time_column = None
            self.signal_columns = []
            self.selected_signals = []
            self.signal_listbox.delete(0, tk.END)
            self.reference_signal_combobox['values'] = []
            self.reference_signal_var.set('')

    def _get_processed_data(self, signal_col, full_data=True):
        """
        Returns the processed (normalized or original) data for a given signal column.
        If full_data is False, returns decimated data for plotting.
        """
        if self.data is None or signal_col not in self.data.columns:
            return np.array([])

        data_series = self.data[signal_col].values
        
        # Apply normalization first
        if self.normalize_var.get():
            ref_signal_name = self.reference_signal_var.get()
            if ref_signal_name and ref_signal_name in self.data.columns:
                reference_data = self.data[ref_signal_name].values
                # Avoid division by zero
                reference_data[reference_data == 0] = 1e-9 # Small number to prevent division by zero
                data_series = data_series / reference_data
            else:
                messagebox.showwarning("Normalization Warning", f"Reference signal '{ref_signal_name}' not found or not selected. Normalization skipped.")
                self.normalize_var.set(False) # Uncheck normalize if reference is invalid

        # Then apply detrending
        if self.detrend_var.get():
            try:
                #we try the savitzky golay filter or detrending
                #dt = np.mean(np.diff(self.data[self.time_column].values))
                #window_time_length = 0.1 # seconds
                #window_length = int(window_time_length/dt) # window length must be odd and less than data length
                #data_series = savgol_filter(data_series, window_length=window_length, polyorder=3)
                
                data_series = detrend(data_series, type='constant')
            except Exception as e:
                messagebox.showwarning("Detrending Warning", f"Could not detrend signal '{signal_col}': {e}. Detrending skipped.")
                self.detrend_var.set(False) # Uncheck detrend if it fails

        calibration_factor = 1.0
        if self.calibrate_var.get():
            idx = list(self.data.columns).index(signal_col) - 1
            calibration_factor = self.CALIBRATION_FACTORS[idx]
        

        if not full_data: # Decimate for plotting
            decimation_factor = 50 # Changed from 10 to 50
            return data_series[::decimation_factor] * calibration_factor
        
        return data_series * calibration_factor

    def on_normalize_toggle(self):
        """Callback for toggling signal normalization; replot selected signals."""
        # Re-plot when normalize checkbox is toggled
        self.plot_selected_signals()

    def on_detrend_toggle(self): # New handler for detrend checkbox
        """Callback for toggling signal detrending; replot selected signals."""
        # Re-plot when detrend checkbox is toggled
        self.plot_selected_signals()

    def on_calibrate_toggle(self): # New handler for detrend checkbox
        """Callback for toggling calibration; replot selected signals."""
        # Re-plot when detrend checkbox is toggled
        self.plot_selected_signals()

    def on_reference_signal_select(self, event):
        """Callback for changing the reference signal used for normalization.
        
        Triggers a replot of the currently selected signals.
        """
        # Re-plot when reference signal is changed
        self.plot_selected_signals()

    def plot_all_signals(self):
        """Plot the first seven signal channels vs. time on the main axes.
        
        Uses decimated, preprocessed data for a quick overview of the dataset.
        """
        if self.data is None or self.time_column is None:
            return

        self.ax.cla() # Clear the single axes
        t = self.data[self.time_column].values
        
        # Decimate data for plotting to improve performance
        decimation_factor = 50 # Plot every 10th point
        t_decimated = t[::decimation_factor]
        
        # Plotting the first 7 signals as per user's description
        signals_to_plot = self.signal_columns[:7] 
        
        for signal_col in signals_to_plot:
            # Use _get_processed_data for plotting (decimated)
            y_decimated = self._get_processed_data(signal_col, full_data=False)
            self.ax.plot(t_decimated, y_decimated, label=signal_col)
        
        self.ax.set_title("Signals vs Time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def update_conversion_factors_from_cf(self, event):
        """Update QPD calibration factors when the CF field is edited.
        
        Recomputes the conversion factors for the QPD channels and refreshes the
        plot so that the new physical scaling is visible.
        """
        # called when cf_entry gets modified
        self.CALIBRATION_FACTORS[2] = self.cf_var.get() * 1000.0
        self.CALIBRATION_FACTORS[3] = self.cf_var.get() * 1000.0

        # replot the signals
        self.plot_selected_signals()

    def on_signal_select(self, event):
        """Update the list of selected signals from the listbox and replot."""
        # Update the list of selected signals
        selected_indices = self.signal_listbox.curselection()
        self.selected_signals = [self.signal_listbox.get(i) for i in selected_indices]
        # Re-plot immediately when selection changes
        self.plot_selected_signals()

    def plot_selected_signals(self):
        """Plot the currently selected signals vs. time on the main axes.
        
        Clears previous contents, applies preprocessing (normalize, detrend,
        calibrate) to each selected signal, plots them versus time, and restores
        any existing time-selection highlight.
        """
        if self.data is None or self.time_column is None or not self.selected_signals:
            return

        self.ax.cla() # Clear the single axes
        # Reset line markers and patches when axes are cleared
        self.start_line = None
        self.end_line = None
        for patch in self.highlight_patches:
            try: # Added try-except for robustness
                patch.remove()
            except NotImplementedError:
                pass
        self.highlight_patches = []

        t = self.data[self.time_column].values
        
        # Decimate time data for plotting
        decimation_factor = 50 # Plot every 10th point
        t_decimated = t[::decimation_factor]

        plotted_signals_count = 0
        last_plotted_signal = None
        
        for signal_col in self.selected_signals:
            # Use _get_processed_data for plotting (decimated)
            y_decimated = self._get_processed_data(signal_col, full_data=False)
            self.ax.plot(t_decimated, y_decimated, label=signal_col)
            plotted_signals_count += 1
            last_plotted_signal = signal_col
        
        self.ax.set_title("Selected Signals vs Time")
        self.ax.set_xlabel("Time (s)")

        if (plotted_signals_count != 1) or (not self.calibrate_var.get()) :
            y_title = "Voltage (V)"
        else:
            idx = list(self.data.columns).index(last_plotted_signal) - 1
            y_title = f'{self.SIGNAL_TITLES[idx]} ({self.CALIBRATION_UNITS[idx]})'

        
        self.ax.set_ylabel(y_title)

        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        # FIX: Redraw time selection if it exists
        if self.selected_time_range and self.selected_time_range[0] is not None and self.selected_time_range[1] is not None:
            start_t, end_t = sorted(self.selected_time_range)
            self.start_line = self.ax.axvline(start_t, color='red', linestyle='--', lw=1, label='Selection Start')
            self.end_line = self.ax.axvline(end_t, color='green', linestyle='--', lw=1, label='Selection End')
            highlight_patch = self.ax.axvspan(start_t, end_t, color='yellow', alpha=0.3, label='Selected Time Range')
            self.highlight_patches.append(highlight_patch)
            self.fig.canvas.draw_idle()

    # --- Time Selection Logic ---
    def on_plot_click(self, event):
        """Handle mouse clicks on the plot to define or reset the time range.
        
        Left clicks set start and end times and draw vertical markers; right
        clicks clear the current selection and remove all markers.
        """
        if event.inaxes != self.ax: return
        if event.button == 1: # Left mouse button
            if self.selected_time_range is None:
                # First click, store start time
                self.selected_time_range = [event.xdata, None]
                self.draw_selection_marker(event.xdata, 'start')
            elif self.selected_time_range[1] is None:
                # Second click, store end time
                self.selected_time_range[1] = event.xdata
                self.draw_selection_marker(event.xdata, 'end')
                self.draw_time_range_highlight()
                # Optionally, auto-compute PSD or save data after second click
                # For now, we just highlight and wait for button press
            else:
                # Third click, reset selection
                self.reset_time_selection()
                self.selected_time_range = [event.xdata, None]
                self.draw_selection_marker(event.xdata, 'start')
        elif event.button == 3: # Right mouse button, reset selection
            self.reset_time_selection()

    def on_plot_release(self, event):
        """Placeholder for mouse button release logic (currently not used)."""
        # This could be used for drag-selection, but for simplicity, we'll stick to two clicks.
        pass

    def on_plot_motion(self, event):
        """Placeholder for mouse-motion logic during selection (currently unused)."""
        # Optional: show a vertical line following the cursor
        if event.inaxes == self.ax and self.selected_time_range and self.selected_time_range[1] is None:
            # Draw a temporary line for the second click preview
            pass # This can get complex with clearing and redrawing

    def draw_selection_marker(self, x_val, type):
        """Draw a vertical line marking the start or end of the selected time window."""
        # Draw a vertical line to mark the selection point
        if type == 'start':
            self.start_line = self.ax.axvline(x_val, color='red', linestyle='--', lw=1, label='Selection Start')
        elif type == 'end':
            self.end_line = self.ax.axvline(x_val, color='green', linestyle='--', lw=1, label='Selection End')
        self.fig.canvas.draw_idle()

    def draw_time_range_highlight(self):
        """Highlight the selected time interval as a shaded region on the plot."""
        if self.selected_time_range and self.selected_time_range[0] is not None and self.selected_time_range[1] is not None:
            start_t, end_t = sorted(self.selected_time_range)
            # Highlight the selected region on the plot
            highlight_patch = self.ax.axvspan(start_t, end_t, color='yellow', alpha=0.3, label='Selected Time Range')
            self.highlight_patches.append(highlight_patch) # Store the patch
            self.fig.canvas.draw_idle()

    def reset_time_selection(self):
        """Clear all time-selection markers and shaded regions and reset the selection."""
        # Safeguard against removing artists that are None or already removed
        try:
            if self.start_line: # Check if start_line exists and is not None
                self.start_line.remove()
                self.start_line = None
        except NotImplementedError:
            self.start_line = None # Ensure it's None if removal failed
        
        try:
            if self.end_line: # Check if end_line exists and is not None
                self.end_line.remove()
                self.end_line = None
        except NotImplementedError:
            self.end_line = None # Ensure it's None if removal failed
        
        # Remove all stored highlight patches
        for patch in self.highlight_patches:
            try:
                patch.remove()
            except NotImplementedError:
                pass # Ignore if already removed or invalid
        self.highlight_patches = [] # Clear the list
        
        self.selected_time_range = None
        self.fig.canvas.draw_idle()

    # --- Action Functions ---
    def compute_psd(self):
        """Compute and display PSD for selected signals over the chosen time range.
        
        Uses scipy.signal.welch() to estimate the power spectral density for
        each selected signal, creates a new figure with the PSD curves, and
        stores the PSD of the first selected signal for optional export.
        """
        if self.data is None or not self.selected_signals or self.selected_time_range is None or self.selected_time_range[1] is None:
            messagebox.showwarning("Selection Error", "Please load data, select signals, and define a time range first.")
            return

        start_t, end_t = sorted(self.selected_time_range)
        
        # Filter data for the selected time range
        time_data = self.data[self.time_column].values
        # Use original time data for computation
        data_in_range_full = self.data[(time_data >= start_t) & (time_data <= end_t)]

        if data_in_range_full.empty:
            messagebox.showerror("Error", "No data found within the selected time range.")
            return

        # Calculate sampling frequency (important for PSD)
        # Use the original time data to get a more stable sampling frequency estimate
        sampling_freq = 1.0 / np.mean(np.diff(self.data[self.time_column].values))
        
        # Create a new figure for PSD plots
        psd_fig, psd_axes = plt.subplots(len(self.selected_signals), 1, figsize=(8, 3 * len(self.selected_signals)))
        if len(self.selected_signals) == 1: # Ensure psd_axes is iterable even for one signal
            psd_axes = [psd_axes]

        for i, signal_col in enumerate(self.selected_signals):
            # Use _get_processed_data for computation (full data)
            signal_data = self._get_processed_data(signal_col, full_data=True)[(time_data >= start_t) & (time_data <= end_t)]
            
            # Compute PSD using Welch's method
            # nperseg: length of each segment. A common choice is 256 or 512.
            # fs: sampling frequency
            try:
                freqs, psd = welch(signal_data, fs=sampling_freq, nperseg=1024) # Increased nperseg for better resolution
                # FIX: Use loglog for both axes logarithmic scale
                psd_axes[i].loglog(freqs, psd) # Changed from semilogy to loglog
                psd_axes[i].set_title(f"PSD of {signal_col}")
                psd_axes[i].set_xlabel("Frequency (Hz)")
                psd_axes[i].set_ylabel("Power Spectral Density (V^2/Hz)")
                psd_axes[i].grid(True)
            except Exception as e:
                messagebox.showerror("PSD Error", f"Could not compute PSD for {signal_col}: {e}")
                psd_axes[i].set_title(f"PSD Error for {signal_col}")

            # Store the last computed PSD data for saving
            # This assumes we want to save the PSD of the first selected signal
            if i == 0:
                self.last_computed_freqs = freqs
                self.last_computed_psd = psd
                self.last_computed_signal_name = signal_col

        psd_fig.tight_layout()
        # FIX: Make PSD window non-modal so user can interact with main GUI
        plt.show(block=False) # Changed from plt.show() to plt.show(block=False)

    def compute_histogram_and_plots(self):
        """Compute and plot PDFs (histograms) of selected signals in the time window.
        
        Builds normalized histograms in physical units (nm), shows them on
        linear and semilogarithmic plots, optionally computes variance and
        trap stiffness for 'free' data, and stores histogram data for combined
        plotting across files.
        """
        if self.data is None or not self.selected_signals or self.selected_time_range is None or self.selected_time_range[1] is None:
            messagebox.showwarning("Selection Error", "Please load data, select signals, and define a time range first.")
            return

        start_t, end_t = sorted(self.selected_time_range)
        
        # Filter data for the selected time range
        time_data = self.data[self.time_column].values
        # Use original time data for computation
        data_in_range_full = self.data[(time_data >= start_t) & (time_data <= end_t)]

        if data_in_range_full.empty:
            messagebox.showerror("Error", "No data found within the selected time range.")
            return

        # Calculate sampling frequency (important for PSD)
        # Use the original time data to get a more stable sampling frequency estimate
        sampling_freq = 1.0 / np.mean(np.diff(self.data[self.time_column].values))

        # Create a new figure for PDF plots
        hist_fig, hist_axes = plt.subplots(len(self.selected_signals), 1, figsize=(8, 3 * len(self.selected_signals)))
        if len(self.selected_signals) == 1: # Ensure psd_axes is iterable even for one signal
            hist_axes = [hist_axes]

        filename = self.file_path.get()

        # Define gaussian function for fitting
        def gaussian(x, mu, sigma, A, B):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B

        
        for i, signal_col in enumerate(self.selected_signals):
            # Use _get_processed_data for computation (full data)
            signal_data = self._get_processed_data(signal_col, full_data=True)[(time_data >= start_t) & (time_data <= end_t)]
            
            try:
                # Histogram
                hist, bin_edges = np.histogram(signal_data, bins=self.HIST_BINS, range=self.BIN_RANGE, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                

                # normal histogram plot
                hist_axes[i].plot(bin_centers, hist, label='PDF')
                hist_axes[i].set_xlabel("Displacement (nm)")
                hist_axes[i].set_ylabel("Probability Density")
                hist_axes[i].set_title(f"PDF of {signal_col}: {filename}")
                hist_axes[i].grid(True)
                hist_axes[i].set_xlim(self.BIN_RANGE)

                #perform fitting with gaussian
                try:
                    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[0, 1, 1, 0])
                    fitted_hist = gaussian(bin_centers, *popt)
                    hist_axes[i].plot(bin_centers, fitted_hist, 'r--', label='Gaussian Fit')
                    hist_axes[i].legend()

                    A = popt[2]
                    B = popt[3]
                    mu = popt[0]
                    sigma = popt[1]
                    hist_axes[i].annotate(f"Fit: {A:.2f} exp(-(x - {mu:.2f})**2/{sigma:.2f}/2) + {B:.2f}",
                        xy=(0.05, 0.05), xycoords='axes fraction')
                except Exception as e:
                    print(f"Gaussian fit failed for {signal_col}: {e}")

                if 'free' in filename.lower():
                    variance = np.var(signal_data)
                    stiffness = self.K_BOLTZMANN / variance
                    hist_axes[i].annotate(f"Variance: {variance:.2f} nm²\nk: {stiffness:.4f} pN/nm",
                        xy=(0.05, 0.85), xycoords='axes fraction')

            except Exception as e:
                messagebox.showerror("Histogram Error", f"Could not compute histogram for {signal_col}: {e}")
                hist_axes[i].set_title(f"Histogram Error for {signal_col}")


        hist_fig.tight_layout()
        # FIX: Make PSD window non-modal so user can interact with main GUI
        plt.show(block=False) # Changed from plt.show() to plt.show(block=False)


         # Create a new figure for log log PDF plots
        hist_fig, hist_axes = plt.subplots(len(self.selected_signals), 1, figsize=(8, 3 * len(self.selected_signals)))
        if len(self.selected_signals) == 1: # Ensure psd_axes is iterable even for one signal
            hist_axes = [hist_axes]

        
        for i, signal_col in enumerate(self.selected_signals):
            # Use _get_processed_data for computation (full data)
            signal_data = self._get_processed_data(signal_col, full_data=True)[(time_data >= start_t) & (time_data <= end_t)]
            
            try:
                # Histogram
                hist, bin_edges = np.histogram(signal_data, bins=self.HIST_BINS, range=self.BIN_RANGE, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                # normal histogram plot
                hist_axes[i].semilogy(bin_centers, hist + 1e-10, label='PDF')
                hist_axes[i].set_xlabel("Displacement (nm)")
                hist_axes[i].set_ylabel("Probability Density")
                hist_axes[i].set_title(f"PDF of {signal_col}: {filename}")
                hist_axes[i].grid(True)
                hist_axes[i].set_xlim(self.BIN_RANGE)

                if 'free' in filename.lower():

                    variance = np.var(signal_data)
                    stiffness = self.K_BOLTZMANN / variance
                    hist_axes[i].annotate(f"Variance: {variance:.2f} nm²\nk: {stiffness:.4f} pN/nm",
                        xy=(0.05, 0.85), xycoords='axes fraction')
                    
                if i==0:
                    # Store for combined plots
                    self.data_list.append({
                        "label": "Free" if "free" in filename.lower() else "Cell",
                        "bins": bin_centers,
                        "hist": hist,
                        "filename": filename
                    })

            except Exception as e:
                messagebox.showerror("Histogram Error", f"Could not compute histogram for {signal_col}: {e}")
                hist_axes[i].set_title(f"Histogram Error for {signal_col}")


            


        hist_fig.tight_layout()
        # FIX: Make PSD window non-modal so user can interact with main GUI
        plt.show(block=False) # Changed from plt.show() to plt.show(block=False)

    
    def plot_combined_pdf(self):
        """Plot a combined PDF from histogram data collected across multiple files.
        
        Overlays all stored PDFs in a single figure for visual comparison.
        """
        if len(self.data_list) < 2:
            return
        else:
            n_files = len(self.data_list)
        

         # Create a new figure for log log PDF plots
        hist_fig, hist_axes = plt.subplots(1, 1, figsize=(8, 3))
        for data in self.data_list:
            hist_axes.plot(data["bins"], data["hist"], label=data["label"])

        hist_axes.set_xlabel("Displacement (nm)")
        hist_axes.set_ylabel("Probability Density")
        hist_axes.set_title("Combined Log PDF")
        hist_axes.set_xlim(self.BIN_RANGE)
        hist_axes.grid(True, which="both")
        hist_axes.legend()
        #plt.savefig(os.path.join(output_dir, "combined_log_pdf.png"), dpi=300)
        #plt.close()

        hist_fig.tight_layout()
        plt.show(block=False) # Changed from plt.show() to plt.show(block=False)

    def save_selected_data(self):
        """Save the selected time window of the chosen signals to a CSV file."""
        if self.data is None or not self.selected_signals or self.selected_time_range is None or self.selected_time_range[1] is None:
            messagebox.showwarning("Selection Error", "Please load data, select signals, and define a time range first.")
            return

        start_t, end_t = sorted(self.selected_time_range)
        
        # Filter data for the selected time range
        time_data = self.data[self.time_column].values
        data_to_save_dict = {self.time_column: time_data[(time_data >= start_t) & (time_data <= end_t)]}

        for signal_col in self.selected_signals:
            # Use _get_processed_data for saving (full data)
            processed_signal_data = self._get_processed_data(signal_col, full_data=True)[(time_data >= start_t) & (time_data <= end_t)]
            data_to_save_dict[signal_col] = processed_signal_data

        data_to_save = pd.DataFrame(data_to_save_dict)

        if data_to_save.empty:
            messagebox.showerror("Error", "No data found within the selected time range.")
            return

        # Prompt user for save location and filename
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            title="Save Selected Data As"
        )

        if save_path:
            try:
                data_to_save.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Selected data saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save data: {e}")

    def save_psd_data(self):
        """Save the last computed PSD (first selected signal) to a CSV file."""
        if self.last_computed_freqs is None or self.last_computed_psd is None:
            messagebox.showwarning("No PSD Data", "Please compute PSD first.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            title="Save PSD Data As"
        )

        if save_path:
            try:
                psd_data = pd.DataFrame({
                    'Frequency (Hz)': self.last_computed_freqs,
                    f'PSD ({self.last_computed_signal_name})': self.last_computed_psd
                })
                psd_data.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"PSD data saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save PSD data: {e}")

def main():
    """Entry point for running the DataAnalyzerApp as a standalone GUI program."""
    root = tk.Tk()
    app = DataAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()