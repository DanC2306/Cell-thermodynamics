import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from typing import Self, List, Tuple, Dict
from tkinter import filedialog, ttk, messagebox

# for profiling purposes
import cProfile
import pstats

import io
# end for profiling purposes

class DataAnalyzerFptWindow(tk.Toplevel):
    def __init__(self:Self, parent:tk.Tk, time_data:np.ndarray, signal_data:np.ndarray, constants=None)->None:
        super().__init__(parent)
        self.title("FPT Data Analyzer")
        self.geometry("1600x960")
        self.time_data = time_data
        self.signal_data = signal_data
        self.constants = constants
        self.create_widgets()

    def create_widgets(self:Self)->None:

        # Panedwindow for left (controls) and right (plot)
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Create a frame for the controls
        control_frame = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(control_frame, weight=1)

        data_shown_frame = ttk.LabelFrame(control_frame, text="Data Shown Signal", padding="5")
        data_shown_frame.pack(fill=tk.X, padx=5, pady=5)

        data_shown_label = ttk.Label(data_shown_frame, text="Data type:")
        data_shown_label.pack(anchor=tk.W, padx=5, pady=5)
        #data_list_widget = tk.Listbox(data_shown_frame, values=["Raw Signal", "FPGA Signal"], state="readonly")
        self.signal_listbox = tk.Listbox(data_shown_frame, selectmode=tk.SINGLE, exportselection=False)
        self.signal_listbox.insert(tk.END, "Preprocessed Signal")
        self.signal_listbox.insert(tk.END, "FPGA Signal")
        self.signal_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.signal_listbox.bind('<<ListboxSelect>>', self.on_signal_select)
        self.signal_listbox.selection_set(0)  # Select the first item by default


        # Load Data Button
        load_button = ttk.Button(control_frame, text="Load FPGA Data", command=self.load_data)
        load_button.pack(fill=tk.X, padx=5, pady=5)

        user_parameter_frame = ttk.LabelFrame(control_frame, text="User parameters", padding="5")
        user_parameter_frame.pack(fill=tk.X, padx=5, pady=5)

        setpoint_label = ttk.Label(user_parameter_frame, text="Setpoint (% of k_B * T):")
        setpoint_label.pack(anchor=tk.W, padx=5, pady=5)

        self.setpoint_var = tk.DoubleVar(value=0.5)
        self.setpoint_entry = ttk.Entry(user_parameter_frame, textvariable=self.setpoint_var)
        self.setpoint_entry.pack(side=tk.LEFT,fill=tk.X, padx=5, pady=5)


        # Create a frame for the plots
        self.plot_frame = ttk.Frame(self.paned_window, padding="10")
        self.paned_window.add(self.plot_frame, weight=3)

        # Create a Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)    
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        
        self.update_plot(self.time_data, self.signal_data, title="Preprocessed Signal")

    def get_FPT_PDF_fromFPGA(self, fileName):
        fpt = np.array([])  # Placeholder for FPT values
        pdf = np.array([])  # Placeholder for PDF values    
        setpoint_energy = 0.05  # Placeholder for setpoint energy in k_B * T
        return fpt, pdf, setpoint_energy

    def load_data(self:Self)->None:
        file_path = filedialog.askopenfilename(title="Select FPGA Data File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            try:
                fpt, pdf, setpoint_energy =self.get_FPT_PDF_fromFPGA(file_path)
                self.plot_data(fpt, pdf, title="FPGA Data", xlabel="First Passage Time (s)", ylabel="PDF")
                reference_value = np.mean(self.signal_data)

                #update the setpoint entry with the value from the FPGA data
                kBT_setpoint = self.convert_setpoint_from_kBT_to_percentage(reference_value, setpoint_energy)
                self.setpoint_var.set(kBT_setpoint / 100)
                #self.plot_data()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")

    def on_signal_select(self:Self, event)->None:
        selection = self.signal_listbox.curselection()
        if selection:
            selected_signal = self.signal_listbox.get(selection[0])
            if selected_signal == "Preprocessed Signal":
                self.update_plot(self.time_data, self.signal_data, title="Preprocessed Signal")
            elif selected_signal == "FPGA Signal":
                if hasattr(self, 'fpga_time_data') and hasattr(self, 'fpga_signal_data'):
                    self.update_plot(self.fpga_time_data, self.fpga_signal_data, title="FPGA Signal")
                else:
                    messagebox.showwarning("Data Not Loaded", "Please load FPGA data to view this signal.", parent=self)

    def update_plot(self:Self, time_data:np.ndarray, signal_data:np.ndarray, title:str="Signal Plot")->None:
        # this method calculated FPT and updates the plot with the FPT results
        # for now it just updates the plot with the provided data

        setpoint_value = self.convert_setpoint_from_kBT_to_percentage(signal_data, self.setpoint_var.get())

        fpt_values, cdf_values = self.FPT_CDF_fromData(signal_data, time_data, setpoint_value)

        self.plot_data(fpt_values, cdf_values, title=title, xlabel="First Passage Time (s)", ylabel="CDF")


    def convert_setpoint_from_kBT_to_percentage(self:Self, x:np.ndarray, kBT_value:float)->float:
        avg_signal = np.mean(x)
        # Assuming kBT_value is a fraction of kBT, the corresponding x is calculated as E = 1/2 * k_trap * x^2 => x = sqrt(2 * E / k_trap) => x = sqrt(2 * (kBT_value * k_B * T) / k_trap)
        setpoint_value = np.sqrt((2 * kBT_value * self.constants.k_B * self.constants.T) / self.constants.k_trap)

        #make setpoint_value a percentage of the average signal value
        setpoint_value = (setpoint_value / avg_signal) * 100

        return setpoint_value
    
    def convert_setpoint_from_percentage_to_kBT(self:Self, reference_value:float, percentage_value:float)->float:
        setpoint_value = (percentage_value / 100) * reference_value
        kBT_value = (setpoint_value ** 2 * self.constants.k_trap) / (2 * self.constants.k_B * self.constants.T)
        return kBT_value
    

    def plot_data(self:Self, x_data:np.ndarray, y_data:np.ndarray, title:str="Signal Plot", xlabel:str="Time", ylabel:str="Signal Value")->None:
        self.ax.clear()
        self.ax.plot(x_data, y_data, label=title)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()
        self.canvas.draw()

    def FPT_CDF_fromData(x, t, setpoint0, setpoint1 = 0, bins = 100, onlyLongTransitions = False)->Tuple[np.ndarray, np.ndarray]:
        # this method calculates the FPT CDF from the provided data
        # it returns the FPT values and the corresponding CDF values

        # find the indices where the signal crosses the setpoint
        crossings = np.where((x[:-1] < setpoint0) & (x[1:] >= setpoint0))[0] + 1
        if onlyLongTransitions:
            crossings = crossings[(crossings > 1) & (crossings < len(x) - 1)]
            crossings = crossings[(x[crossings - 2] < setpoint0) & (x[crossings + 1] >= setpoint0)]

        # calculate the FPT values
        fpt_values = t[crossings]
        fpt_values -= t[0]  # normalize to start at zero

        # calculate the CDF values
        cdf_values = np.arange(1, len(fpt_values) + 1) / len(fpt_values)

        return (fpt_values, cdf_values)

