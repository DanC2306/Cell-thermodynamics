import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from typing import Self, List, Tuple, Dict
from tkinter import filedialog, ttk, messagebox
from data_preprocessing import *
# for profiling purposes
import cProfile
import pstats

import io
# end for profiling purposes

class DataAnalyzerFptWindow(tk.Toplevel):
    def __init__(self:Self, parent:tk.Tk, time_data:np.ndarray, signal_data:np.ndarray, constants:Constants=None)->None:
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

    def bio_variance_x(self, fileName):
        d, _ = DataAnalyzerApp.getCsvTimeseries(fileName)
        x = np.array(d["x"])
        x2 = np.array(d["x^2"])
        configFile = fileName.replace("_bioControllerAcquisition.csv", "_conf.csv")
        configs = DataAnalyzerApp.getCsvConfigurationFile(configFile)
        squareShift = float(configs["squaresShift"]["Parameter value"])
        x2 /= (2**squareShift)
        return np.mean(x2) - np.mean(x)**2
        
    def fptSetpointToEnergyFraction(self, fileName, setpoint):
        '''knowing that the thermal energy of the particle is E_T = k_b T / 2, 
        and that the elastic energy is E_e = K * x^2 / 2, we can write
        K = k_b T / σ^2_x, thus
        E_e = k_b T / 2 * x^2 / σ^2_x = E_T * x^2 / σ^2_x
        So, if the particle is at a certain distance from the center, we 
        can correlate its elastic energy to its intrinsic thermal energy.

        Also, this correlation is valid even if we don't have the correct 
        scale of x. Assuming to have access to the signal v, where x = a*v 
        (but we don't know the value of a), we have that 
        σ^2_x = E[x^2] - E[x]^2 = E[a^2v^2] - E[av]^2 = a^2(E[v^2]-E[v]^2) = a^2 σ^2_v
        and x^2 = a^2v^2, thus x^2 / σ^2_x = a^2 v^2 / a^2σ^2_v = v^2 / σ^2_v
        '''
        # setpoint = float(self.configurations["binFeedback_x0"]["Parameter value"])
        fileName = fileName.replace("_bioControllerTimings.csv", "_bioControllerAcquisition.csv")
        return setpoint**2 / self.bio_variance_x(fileName)

    def get_FT_CDF_fromFPGA(self, fileName):
        crossings, _ = DataAnalyzerApp.getCsvTimeseries(fileName)
        
        t = np.array(crossings['timing'])
        reachedThresholds = np.array(crossings["reachedThreshold"])

        transitionIndexes = 1+np.where(reachedThresholds[1:]!=reachedThresholds[:-1])[0]#where we have a transition
        #elements after the last transition are not usable, because we don't have info on when the next cross would have been
        if transitionIndexes.size == 0:
            return np.array([])
        lastUsableIndex = transitionIndexes[-1]
        reachedThresholds = reachedThresholds[:lastUsableIndex+1]
        t = t[:lastUsableIndex+1]
        longestTimes=t[transitionIndexes]
        fpt = longestTimes[reachedThresholds[transitionIndexes] == 1]
        fpt = np.sort(fpt)
        cdf = np.linspace(0,1,len(fpt))        
        unique, count = np.unique(fpt, return_counts=True)			
        uniqueIndexes = np.concatenate(([0],np.cumsum(count)[:-1]))
        cdf[:len(unique)] = cdf[uniqueIndexes]
        fpt[:len(unique)] = unique
        cdf[len(unique):] = 1
        fpt[len(unique):] = unique[-1]
        configFile = fileName.replace("_bioControllerTimings.csv", "_conf.csv")
        configs = DataAnalyzerApp.getCsvConfigurationFile(configFile)
        setpoint_adimensional = float(configs["binFeedback_x0"]["Parameter value"])
        setpoint_energy = self.convert_setpoint_from_adimensionalDistane_to_kBT(1, setpoint_adimensional)
        return fpt, cdf, setpoint_energy

    def load_data(self:Self)->None:
        file_path = filedialog.askopenfilename(title="Select FPGA Data File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            try:
                fpt, pdf, setpoint_energy =self.get_FT_CDF_fromFPGA(file_path)
                self.plot_data(fpt, pdf, title="FPGA Data", xlabel="First Passage Time (s)", ylabel="PDF")
                reference_value = np.mean(self.signal_data)

                #update the setpoint entry with the value from the FPGA data
                kBT_setpoint = self.convert_setpoint_from_kBT_to_adimensionalDistane(reference_value, setpoint_energy)
                self.setpoint_var.set(kBT_setpoint)
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

        setpoint_value = self.convert_setpoint_from_kBT_to_adimensionalDistane(signal_data, self.setpoint_var.get())

        fpt_values, cdf_values = self.FPT_CDF_fromData(signal_data, time_data, setpoint_value)

        self.plot_data(fpt_values, cdf_values, title=title, xlabel="First Passage Time (s)", ylabel="CDF")


    def convert_setpoint_from_kBT_to_adimensionalDistane(self:Self, x:np.ndarray, kBT_value:float)->float:
        # Assuming kBT_value is a fraction of kBT, the corresponding x is calculated as E = 1/2 * k_trap * x^2 => x = sqrt(2 * E / k_trap) => x = sqrt(2 * (kBT_value * k_B * T) / k_trap)
        setpoint_value = np.sqrt((2 * kBT_value * self.constants.k_B * self.constants.T) / self.constants.k_trap)

        return setpoint_value
    
    def convert_setpoint_from_adimensionalDistane_to_kBT(self:Self, reference_value:float, adimensionalDistane:float)->float:
        setpoint_value = adimensionalDistane * reference_value
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

    @staticmethod
    def FPT_CDF_fromData(x, t, setpoint0, setpoint1 = 0, bins = 100):
        '''
        returns the probability distribution function (PDF) of the first passage time (FPT) of a signal x, with the corresponding times.
        
        x: signal on which the first passage time is calculated

        t: corresponding timings of the signal

        setpoint0: inital setpoint. The first passage times will start when x crosses this value. This variable 
            can be a vector of values. In that case, the FPT will be calculated for each specified setpoint, 
            returning a grid of values

        setpoint1: final setpoint. The first passage times will end when x crosses this value
        
        bins: number of points of the obtained PDF

        Returns:
            if setpoint1 is a single values
                (fpt, pdf): the first passage time and the corresponding probability density function (PDF) (pdf[i] = Probability(first passage time(setpoint1,setpoint0) == fpt[i]))
            if setpoint1 is a list of values
                (fptGrid, setpoint1Grid, pdf): The grid values for the first passage time and setpoint, and the corresponding PDF (pdf[i, j] = Probability(first passage time(j,setpoint0) == fpt[i]))
        '''
        #stupid floats, differences that should be the same are not the same, let's just work with integers first and then rescale them at the end
        dt = t[1]-t[0]
        t = np.round(t/dt).astype(int)

        x0, x1 = setpoint0, setpoint1
        deList = not isinstance(x0, (list, np.ndarray))
        if deList:
            x0 = np.array([x0])
        nOfSetpoints = len(x0)
        x0 = np.array(x0)[None,:]
        x0Crosses = (x[:-1, None] - x0) * (x[1:, None] - x0) <= 0
        x1Crosses = (x[:-1] - x1) * (x[1:] - x1) <= 0

        startIndexes, startSetpointIndexes = np.where(x0Crosses)
        endIndexes = np.where(x1Crosses)[0]
        startSetpointIndexes = startSetpointIndexes[startIndexes < endIndexes[-1]]
        startIndexes = startIndexes[startIndexes < endIndexes[-1]]
        startIdxPositionInEndIndices = np.searchsorted(endIndexes, startIndexes, side = 'right')
       
        transitionTimes = t[endIndexes[startIdxPositionInEndIndices]] - t[startIndexes]
        transitionTimes = transitionTimes.astype(float) * dt#rescale back to float
        pdf = np.zeros((nOfSetpoints, bins))
        fpt = np.zeros((nOfSetpoints, bins))
        cdf = np.linspace(0, 1, bins)
        for i in range(nOfSetpoints):
            currentTransitionTimes = transitionTimes[startSetpointIndexes == i]
            currentTransitionTimes = np.concatenate(([0], np.sort(currentTransitionTimes)))
            fpt[i] = currentTransitionTimes[np.linspace(0, len(currentTransitionTimes)-1, bins, dtype=int)]
            unique, count = np.unique(fpt[i], return_counts=True)			
            uniqueIndexes = np.concatenate(([0],np.cumsum(count)[:-1]))
            pdf[i][:len(unique)] = cdf[uniqueIndexes]
            fpt[i][:len(unique)] = unique
            pdf[i][len(unique):] = 1
            fpt[i][len(unique):] = unique[-1]
            
        if deList:
            return fpt[0], pdf[0]
        startPoints = np.repeat(x0[0,:,None], bins, axis=1)
        return fpt, startPoints, pdf


if __name__ == "__main__":
    t = np.linspace(0,1,500000)
    x = np.random.randn(len(t))
    x = np.convolve(x, np.repeat(1/30,30), mode='same')
    fpt, cdf = DataAnalyzerFptWindow.FPT_CDF_fromData(x, t,.15)
    plt.plot(fpt, cdf)
    fpt, cdf = DataAnalyzerFptWindow.FPT_CDF_fromData(x, t,.1)
    plt.plot(fpt, cdf)
    plt.show()
