import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
import threading

# Add current directory to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import FactorModel, PopulationGenerator
from src.utils import create_simple_structure
import numpy as np

class PsychometricGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Psychometric Population Generator")
        self.root.geometry("500x600")
        
        # Structure Input
        ttk.Label(root, text="Factor Structure (items per factor):").pack(pady=(10, 0))
        ttk.Label(root, text="e.g., '5,10,5' for 3 factors", font=("Arial", 8)).pack(pady=(0, 5))
        self.structure_var = tk.StringVar(value="5,5,5")
        ttk.Entry(root, textvariable=self.structure_var, width=40).pack()
        
        # Reliability Input
        ttk.Label(root, text="Target Reliability (0.0 - 1.0):").pack(pady=(10, 0))
        self.reliability_var = tk.DoubleVar(value=0.8)
        self.reliability_scale = ttk.Scale(root, from_=0.0, to=1.0, variable=self.reliability_var, orient='horizontal', length=300)
        self.reliability_scale.pack()
        self.reliability_label = ttk.Label(root, text="0.80")
        self.reliability_label.pack()
        self.reliability_scale.bind("<Motion>", self.update_reliability_label)
        
        # Likert Scale Input
        ttk.Label(root, text="Likert Scale Points (2-10):").pack(pady=(10, 0))
        self.likert_var = tk.IntVar(value=5)
        self.likert_scale = ttk.Scale(root, from_=2, to=10, variable=self.likert_var, orient='horizontal', length=300)
        self.likert_scale.pack()
        self.likert_label = ttk.Label(root, text="5 Points")
        self.likert_label.pack()
        self.likert_scale.bind("<Motion>", self.update_likert_label)

        # Likert Scale Checkbox
        self.use_likert_var = tk.BooleanVar(value=True)
        self.likert_check = ttk.Checkbutton(root, text="Discretize to Likert Scale", variable=self.use_likert_var, command=self.toggle_likert)
        self.likert_check.pack(pady=(5, 0))

        # Loading Noise Input
        ttk.Label(root, text="Loading Noise (0.0 - 1.0):").pack(pady=(10, 0))
        self.noise_var = tk.DoubleVar(value=0.0)
        self.noise_scale = ttk.Scale(root, from_=0.0, to=1.0, variable=self.noise_var, orient='horizontal', length=300)
        self.noise_scale.pack()
        self.noise_label = ttk.Label(root, text="0.00")
        self.noise_label.pack()
        self.noise_scale.bind("<Motion>", self.update_noise_label)
        self.noise_scale.bind("<Motion>", self.update_noise_label)

        # Mean and Standard Deviation Inputs
        # Create a frame for these two
        self.pop_params_frame = ttk.LabelFrame(root, text="Population Parameters")
        self.pop_params_frame.pack(pady=(10, 0), padx=10, fill="x")
        
        # Mean
        self.param_grid = ttk.Frame(self.pop_params_frame)
        self.param_grid.pack(pady=5)
        
        ttk.Label(self.param_grid, text="Latent Mean (0=Avg):").grid(row=0, column=0, padx=5)
        self.mean_var = tk.DoubleVar(value=0.0)
        ttk.Entry(self.param_grid, textvariable=self.mean_var, width=10).grid(row=0, column=1, padx=5)
        
        # SD
        ttk.Label(self.param_grid, text="Latent SD (1=Std):").grid(row=0, column=2, padx=5)
        self.std_var = tk.DoubleVar(value=1.0)
        ttk.Entry(self.param_grid, textvariable=self.std_var, width=10).grid(row=0, column=3, padx=5)

        # Sum Score Options
        self.sum_score_frame = ttk.LabelFrame(root, text="Sum Score Options (e.g., IQ)")
        self.sum_score_frame.pack(pady=(10, 0), padx=10, fill="x")
        
        # Checkboxes
        self.calc_sum_var = tk.BooleanVar(value=False)
        self.rescale_sum_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(self.sum_score_frame, text="Calculate Sum Score (Total)", variable=self.calc_sum_var).pack(anchor="w", padx=5)
        ttk.Checkbutton(self.sum_score_frame, text="Rescale Sum Score (Target Mean/SD)", variable=self.rescale_sum_var).pack(anchor="w", padx=5)
        
        # Target Params
        self.target_grid = ttk.Frame(self.sum_score_frame)
        self.target_grid.pack(pady=5)
        
        ttk.Label(self.target_grid, text="Target Mean:").grid(row=0, column=0, padx=5)
        self.target_mean_var = tk.DoubleVar(value=100.0)
        ttk.Entry(self.target_grid, textvariable=self.target_mean_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.target_grid, text="Target SD:").grid(row=0, column=2, padx=5)
        self.target_std_var = tk.DoubleVar(value=15.0)
        ttk.Entry(self.target_grid, textvariable=self.target_std_var, width=10).grid(row=0, column=3, padx=5)

        ttk.Label(root, text="Number of Subjects (N):").pack(pady=(10, 5))
        self.samples_var = tk.IntVar(value=1000)
        ttk.Entry(root, textvariable=self.samples_var, width=20).pack()
        
        # Output File Selection
        ttk.Label(root, text="Output File:").pack(pady=(20, 5))
        self.output_frame = ttk.Frame(root)
        self.output_frame.pack()
        
        self.output_path_var = tk.StringVar(value="population_data.csv")
        ttk.Entry(self.output_frame, textvariable=self.output_path_var, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.output_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        
        # Generate Button
        ttk.Button(root, text="Generate", command=self.generate).pack(pady=30)

        # Custom Loadings
        self.custom_loadings = None
        ttk.Button(root, text="Customize Loadings...", command=self.open_loading_editor).pack(pady=(10, 5))
        
        # Status
        self.status_var = tk.StringVar()
        ttk.Label(root, textvariable=self.status_var, foreground="blue").pack()

    def update_reliability_label(self, event):
        self.reliability_label.config(text=f"{self.reliability_var.get():.2f}")

    def update_likert_label(self, event):
        self.likert_label.config(text=f"{int(self.likert_var.get())} Points")

    def update_noise_label(self, event):
        self.noise_label.config(text=f"{self.noise_var.get():.2f}")

    def toggle_likert(self):
        if self.use_likert_var.get():
            self.likert_scale.state(['!disabled'])
            self.likert_label.config(foreground="black")
        else:
            self.likert_scale.state(['disabled'])
            self.likert_label.config(foreground="gray")

    def browse_file(self):
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.output_path_var.set(filename)

    def open_loading_editor(self):
        # 1. Parse current structure to know dims
        structure_str = self.structure_var.get()
        try:
            items_per_factor = [int(x.strip()) for x in structure_str.split(',')]
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid structure first (e.g., 5,10,5)")
            return

        n_factors = len(items_per_factor)
        n_vars = sum(items_per_factor)
        
        # 2. Create Window
        editor = tk.Toplevel(self.root)
        editor.title("Customize Factor Loadings")
        editor.geometry("600x500")
        
        # Scrollable Frame Pattern
        container = ttk.Frame(editor)
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        container.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 3. Create Grid
        # Header
        ttk.Label(scrollable_frame, text="Item").grid(row=0, column=0, padx=5, pady=5)
        for f in range(n_factors):
            ttk.Label(scrollable_frame, text=f"Factor {f+1}").grid(row=0, column=f+1, padx=5, pady=5)

        # Pre-calculate simple structure for defaults
        defaults = create_simple_structure(
            n_factors, 
            items_per_factor, 
            reliability=self.reliability_var.get()
        )
        
        # Apply current noise setting to defaults for visualization
        current_noise = self.noise_var.get()
        if current_noise > 0:
            noise_matrix = np.random.uniform(-current_noise, current_noise, size=defaults.shape)
            defaults = defaults + noise_matrix
        
        # Helper to populate/refresh grid - NOT USED YET, using inline loop
        # We need a reference to the Entry widgets to update them later
        entry_widgets = []

        # Rows Generation
        for i in range(n_vars):
            ttk.Label(scrollable_frame, text=f"Item {i+1}").grid(row=i+1, column=0, padx=5, pady=2)
            row_widgets = []
            for f in range(n_factors):
                # Init value logic
                if self.custom_loadings is not None and self.custom_loadings.shape == (n_vars, n_factors):
                     val = self.custom_loadings[i, f]
                else:
                     val = defaults[i, f]

                entry = ttk.Entry(scrollable_frame, width=8)
                entry.insert(0, f"{val:.2f}")
                entry.grid(row=i+1, column=f+1, padx=2, pady=2)
                row_widgets.append(entry)
            entry_widgets.append(row_widgets)
            
        # Function to Regenerate
        def regenerate():
            # Recalculate defaults based on CURRENT sliders
            new_defaults = create_simple_structure(
                n_factors, 
                items_per_factor, 
                reliability=self.reliability_var.get()
            )
            # Apply Noise
            c_noise = self.noise_var.get()
            if c_noise > 0:
                n_mat = np.random.uniform(-c_noise, c_noise, size=new_defaults.shape)
                new_defaults = new_defaults + n_mat
            
            # Clip to [-1, 1]
            new_defaults = np.clip(new_defaults, -1.0, 1.0)
            
            # Update all entries
            for i in range(n_vars):
                for f in range(n_factors):
                    entry_widgets[i][f].delete(0, tk.END)
                    entry_widgets[i][f].insert(0, f"{new_defaults[i, f]:.2f}")
            
            messagebox.showinfo("Regenerated", "Values reset based on current sliders (including Noise).")

        # Control Frame
        control_frame = ttk.Frame(editor)
        control_frame.pack(pady=5)
        
        ttk.Button(control_frame, text="Regenerate from Sliders", command=regenerate).pack(side=tk.LEFT, padx=5)

        # Save Button
        def save_matrix():
            try:
                new_matrix = np.zeros((n_vars, n_factors))
                for i in range(n_vars):
                    for f in range(n_factors):
                        val_str = entry_widgets[i][f].get()
                        new_matrix[i, f] = float(val_str)
                
                self.custom_loadings = new_matrix
                messagebox.showinfo("Saved", "Custom loadings saved! Using this exact matrix for generation.")
                editor.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid value in matrix. Must be numbers.")

        ttk.Button(control_frame, text="Save & Close", command=save_matrix).pack(side=tk.LEFT, padx=5)

    def generate(self):
        try:
            # Parse inputs
            structure_str = self.structure_var.get()
            try:
                items_per_factor = [int(x.strip()) for x in structure_str.split(',')]
            except ValueError:
                messagebox.showerror("Input Error", "Structure must be integers separated by commas (e.g., 5,10,5)")
                return
            
            reliability = self.reliability_var.get()
            n_samples = self.samples_var.get()
            output_path = self.output_path_var.get()
            
            # New params
            pop_mean = self.mean_var.get()
            pop_std = self.std_var.get()
            
            # Sum Score Logic
            # If user wants sum score, we calculate it after generation.
            # We don't need the warning anymore because user has specific controls for scaling result.
            
            if not output_path:
                messagebox.showerror("Input Error", "Please specify an output file.")
                return

            self.status_var.set("Generating...")
            self.root.update()
            
            # Logic (using existing utils and model)
            n_factors = len(items_per_factor)
            
            if self.custom_loadings is not None:
                # Validate dimensions matches current structure input
                expected_vars = sum(items_per_factor)
                if self.custom_loadings.shape != (expected_vars, n_factors):
                     if messagebox.askyesno("Warning", "Structure changed since custom loadings passed. Discard custom loadings?"):
                         loadings = create_simple_structure(n_factors, items_per_factor, reliability)
                         self.custom_loadings = None
                         # Re-apply noise to defaults since we reset
                         noise_level = self.noise_var.get()
                         if noise_level > 0:
                             noise_matrix = np.random.uniform(-noise_level, noise_level, size=loadings.shape)
                             loadings = loadings + noise_matrix
                         # Clip defaults
                         loadings = np.clip(loadings, -1.0, 1.0)
                     else:
                         return # Cancel generation
                else:
                    loadings = self.custom_loadings
            else:
                loadings = create_simple_structure(
                    n_factors=n_factors,
                    n_items_per_factor=items_per_factor,
                    reliability=reliability
                )
            
                # Apply Loading Noise (Only if NOT using custom loadings)
                noise_level = self.noise_var.get()
                if noise_level > 0:
                    # Add uniform noise [-noise, +noise]
                    noise_matrix = np.random.uniform(-noise_level, noise_level, size=loadings.shape)
                    loadings = loadings + noise_matrix
                
                # Clip to [-1, 1]
                loadings = np.clip(loadings, -1.0, 1.0)
            
            # Default orthogonal with some correlation if > 1 factor?
            # Let's add standard correlation 0.3 if > 1 factor for realism
            phi = None
            if n_factors > 1:
                phi = np.eye(n_factors)
                phi[phi == 0] = 0.3
            
            model = FactorModel(loadings=loadings, factor_correlations=phi)
            generator = PopulationGenerator(model)
            
            likert = int(self.likert_var.get())
            if not self.use_likert_var.get():
                likert = None
                
            df = generator.generate(n_samples=n_samples, likert_points=likert, mean=pop_mean, std=pop_std)
            
            # Post-processing for Sum Scores
            if self.calc_sum_var.get():
                # sum all columns (assuming all are items)
                df['Total_Score'] = df.sum(axis=1)
                
                if self.rescale_sum_var.get():
                    t_mean = self.target_mean_var.get()
                    t_std = self.target_std_var.get()
                    
                    # Z-score normalization of the sum
                    current_mean = df['Total_Score'].mean()
                    current_std = df['Total_Score'].std()
                    
                    z_scores = (df['Total_Score'] - current_mean) / current_std
                    
                    # Apply target
                    df['Scaled_Score'] = z_scores * t_std + t_mean
                    df['Scaled_Score'] = df['Scaled_Score'].round(2) # clean up
            
            df.to_csv(output_path, index=False)
            
            self.status_var.set(f"Done! Saved to {output_path}")
            messagebox.showinfo("Success", f"Successfully generated {n_samples} subjects.\nSaved to {output_path}")
            
        except Exception as e:
            self.status_var.set("Error!")
            messagebox.showerror("Execution Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = PsychometricGeneratorGUI(root)
    root.mainloop()
