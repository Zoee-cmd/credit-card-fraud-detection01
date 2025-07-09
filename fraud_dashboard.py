#GUI BY ANESU & JELSON
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from datetime import datetime
#Pickle is for loading pre-trained models
import pickle
#for 3D visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class FraudDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("CBZ Bank - Fraud Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f2f5")
        
        # Initialize data
        self.total_transactions = 0
        self.fraudulent_transactions = 0
        
        # Create main container
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Load models
        try:
            with open('models/fraud_detection_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except:
            tk.messagebox.showerror("Error", "Model files not found!")
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create header
        header = ttk.Label(self.main_container, 
                          text="CBZ Bank Fraud Detection System",
                          font=('Helvetica', 16, 'bold'),
                          foreground="#1976d2")
        header.pack(fill=tk.X, pady=(0, 15))
        
        # Create main sections
        self.create_stats_cards()
        self.create_transaction_section()
        self.create_visualization_section()
        self.create_results_section()
        self.create_transactions_table()
        
    def create_stats_cards(self):
        stats_frame = ttk.Frame(self.main_container)
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Statistics cards
        stats = [
            ("Total Transactions", "0", "#1976d2"),
            ("Fraudulent Transactions", "0", "#dc3545"),
            ("Suspicious Transactions", "0", "#ffc107"),
            ("Fraud Detection Rate", "0%", "#28a745")
        ]
        
        for i, (title, value, color) in enumerate(stats):
            card = ttk.Frame(stats_frame, style='Card.TFrame')
            card.grid(row=0, column=i, padx=5, sticky="nsew")
            
            ttk.Label(card, 
                     text=title,
                     font=('Helvetica', 10),
                     style='CardTitle.TLabel').pack(pady=(5, 2))
            
            ttk.Label(card,
                     text=value,
                     font=('Helvetica', 14, 'bold'),
                     foreground=color).pack(pady=(0, 5))
            
        stats_frame.grid_columnconfigure((0,1,2,3), weight=1)
        
    def create_transaction_section(self):
        input_frame = ttk.LabelFrame(self.main_container, 
                                   text="Transaction Analysis",
                                   padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create two columns
        left_frame = ttk.Frame(input_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = ttk.Frame(input_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.transaction_inputs = {}
        
        # Left column inputs
        self.create_input_field(left_frame, "Amount ($):", 'amount', 0)
        self.create_input_field(left_frame, "Transaction Type:", 'type', 1, 
                              combobox=True, values=['Credit', 'Debit'])
        
        # Right column inputs
        self.create_input_field(right_frame, "Location Risk (0-1):", 'location_risk', 0)
        self.create_input_field(right_frame, "Device Risk (0-1):", 'device_risk', 1)
        
        # Analysis button and results
        control_frame = ttk.Frame(input_frame)
        control_frame.pack(side=tk.RIGHT, padx=10)
        
        analyze_btn = ttk.Button(control_frame,
                               text="Analyze Transaction",
                               command=self.analyze_transaction,
                               style='Small.TButton')
        analyze_btn.pack(pady=5)
        
        self.result_var = tk.StringVar(value="Risk Level: -")
        self.result_label = ttk.Label(control_frame,
                                    textvariable=self.result_var,
                                    font=('Helvetica', 12, 'bold'))
        self.result_label.pack(pady=5)
        
    def create_input_field(self, parent, label_text, key, row, combobox=False, values=None):
        ttk.Label(parent, 
                 text=label_text,
                 font=('Helvetica', 9)).grid(row=row, column=0, sticky='w', pady=2)
        
        if combobox:
            self.transaction_inputs[key] = ttk.Combobox(parent, 
                                                      values=values,
                                                      font=('Helvetica', 9),
                                                      state='readonly')
            self.transaction_inputs[key].set(values[0])
        else:
            self.transaction_inputs[key] = ttk.Entry(parent, 
                                                   font=('Helvetica', 9))
        
        self.transaction_inputs[key].grid(row=row, column=1, sticky='ew', padx=(5, 0), pady=2)
        
    def create_visualization_section(self):
        #3D Risk Distributions  by viridis colormap for better visualization
        viz_frame = ttk.Frame(self.main_container)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create 3D risk distribution plot
        risk_frame = ttk.LabelFrame(viz_frame, text="Risk Distribution")
        risk_frame.grid(row=0, column=0, padx=5, sticky="nsew")
        
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate sample data
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        z = np.random.normal(0, 1, 100)
        
        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
        ax.set_title('3D Risk Distribution', fontsize=9)
        
        canvas = FigureCanvasTkAgg(fig, master=risk_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create 3D fraud trends plot
        trends_frame = ttk.LabelFrame(viz_frame, text="Fraud Trends")
        trends_frame.grid(row=0, column=1, padx=5, sticky="nsew")
        
        fig2 = plt.figure(figsize=(5, 3))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # Generate sample data
        x2 = np.linspace(0, 10, 100)
        y2 = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x2, y2)
        Z = np.sin(X) + np.cos(Y)
        
        # Create surface plot
        surf = ax2.plot_surface(X, Y, Z, cmap='viridis')
        ax2.set_title('3D Fraud Trends', fontsize=9)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=trends_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        viz_frame.grid_columnconfigure((0,1), weight=1)
        
    def create_results_section(self):
        results_frame = ttk.LabelFrame(self.main_container,
                                     text="Analysis Results",
                                     padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.fraud_prob_label = ttk.Label(results_frame,
                                        text="Fraud Probability: -",
                                        font=('Helvetica', 10, 'bold'))
        self.fraud_prob_label.pack(side=tk.LEFT, padx=10)
        
        self.risk_factors_label = ttk.Label(results_frame,
                                          text="Risk Factors: -",
                                          font=('Helvetica', 10))
        self.risk_factors_label.pack(side=tk.LEFT, padx=10)
        
    def create_transactions_table(self):
        table_frame = ttk.LabelFrame(self.main_container,
                                   text="Recent Transactions",
                                   padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("ID", "Amount", "Type", "Risk Score", "Status", "Time")
        self.tree = ttk.Treeview(table_frame,
                                columns=columns,
                                show='headings',
                                style='Small.Treeview')
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def analyze_transaction(self):
        try:
            amount = float(self.transaction_inputs['amount'].get())
            transaction_type = 1 if self.transaction_inputs['type'].get() == 'Credit' else 0
            location_risk = float(self.transaction_inputs['location_risk'].get())
            device_risk = float(self.transaction_inputs['device_risk'].get())
            
            if not (0 <= location_risk <= 1 and 0 <= device_risk <= 1):
                raise ValueError("Risk values must be between 0 and 1")
            
            features = {
                'amount': amount,
                'transaction_type': transaction_type,
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'location_risk': location_risk,
                'device_risk': device_risk,
                'time_since_last_transaction': 1.0,
                'avg_transaction_amount': amount,
                'transaction_frequency': 1.0
            }
            
            features_array = np.array(list(features.values())).reshape(1, -1)
            scaled_features = self.scaler.transform(features_array)
            fraud_prob = self.model.predict_proba(scaled_features)[0][1]
            
            self.update_results(features, fraud_prob)
            self.add_transaction(amount, fraud_prob)
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def update_results(self, features, fraud_prob):
        risk_level = "HIGH RISK" if fraud_prob > 0.5 else "LOW RISK"
        self.result_var.set(f"Risk Level: {risk_level}")
        
        self.fraud_prob_label.config(
            text=f"Fraud Probability: {fraud_prob:.1%}",
            foreground='red' if fraud_prob > 0.5 else 'green'
        )
        
        risk_factors = []
        if features['amount'] > 1000: risk_factors.append("High Amount")
        if features['hour'] < 6 or features['hour'] > 22: risk_factors.append("Unusual Time")
        if features['location_risk'] > 0.7: risk_factors.append("Risky Location")
        if features['device_risk'] > 0.7: risk_factors.append("Risky Device")
        
        self.risk_factors_label.config(
            text=f"Risk Factors: {', '.join(risk_factors) if risk_factors else 'None'}"
        )
        
    def add_transaction(self, amount, risk_score):
        transaction_id = f"T{len(self.tree.get_children()) + 1:03d}"
        current_time = datetime.now().strftime("%H:%M:%S")
        
        self.tree.insert("", 0, values=(
            transaction_id,
            f"${amount:.2f}",
            self.transaction_inputs['type'].get(),
            f"{risk_score:.2%}",
            "Suspicious" if risk_score > 0.5 else "Normal",
            current_time
        ))
        
        self.total_transactions += 1
        if risk_score > 0.5:
            self.fraudulent_transactions += 1
        self.update_stats()
        
    def update_stats(self):
        detection_rate = (self.fraudulent_transactions / self.total_transactions * 100) if self.total_transactions > 0 else 0
        
        for card in self.main_container.winfo_children():
            if isinstance(card, ttk.Frame):
                for child in card.winfo_children():
                    if isinstance(child, ttk.Frame):
                        labels = [w for w in child.winfo_children() if isinstance(w, ttk.Label)]
                        if len(labels) >= 2:
                            if "Total Transactions" in labels[0].cget("text"):
                                labels[1].config(text=str(self.total_transactions))
                            elif "Fraudulent Transactions" in labels[0].cget("text"):
                                labels[1].config(text=str(self.fraudulent_transactions))
                            elif "Fraud Detection Rate" in labels[0].cget("text"):
                                labels[1].config(text=f"{detection_rate:.1f}%")

if __name__ == "__main__":
    root = tk.Tk()
    
    style = ttk.Style()
    
    style.configure('Card.TFrame', 
                   background='white',
                   relief='raised',
                   borderwidth=1)
    
    style.configure('Small.TButton',
                   font=('Helvetica', 9, 'bold'),
                   padding=3)
    
    style.configure('Small.Treeview',
                   font=('Helvetica', 9),
                   rowheight=20)
    style.configure('Small.Treeview.Heading',
                   font=('Helvetica', 9, 'bold'))
    
    app = FraudDashboard(root)
    #makes the entire fraud detection dashboard together and makes it functional.
    root.mainloop()
    #Applies to TKinter Application infinite loop keeping application running