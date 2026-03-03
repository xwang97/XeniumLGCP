import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix

# Imports from other modules in the package
from .engine import VoGCAM_SPDE_GPU
from .utils.geometry import barycentric_A
from .utils.spde import spde_precision_with_mass

class XeniumLGCP:
    def __init__(self, 
                 mesh_max_edge=50.0,  # Microns
                 domain_pad=50.0,     # Buffer around tissue
                 kappa=0.2, 
                 rank=20,
                 verbose=True):
        self.mesh_max_edge = mesh_max_edge
        self.domain_pad = domain_pad
        self.kappa = kappa
        self.rank = rank
        self.verbose = verbose
        
        self.df_ = None          
        self.tri_ = None         
        self.coords_ = None      
        self.w_ = None           
        self.Q_base_ = None      
        self.model_ = None       
        self.covariate_name_ = None 

    def load_data(self, file_path):
        if self.verbose: print(f"[IO] Loading {file_path}...")
        self.df_ = pd.read_csv(file_path)
        
        if 'feature_name' not in self.df_.columns and 'gene' in self.df_.columns:
            self.df_.rename(columns={'gene': 'feature_name'}, inplace=True)
            
        req = ['feature_name', 'x_location', 'y_location']
        if not all(c in self.df_.columns for c in req):
            raise ValueError(f"File must contain columns: {req}")
            
        if self.verbose: 
            print(f"[IO] Loaded {len(self.df_):,} transcripts.")
        return self

    def _build_mesh(self):
        if self.df_ is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        all_coords = self.df_[['x_location', 'y_location']].values
        min_x, min_y = all_coords.min(axis=0) - self.domain_pad
        max_x, max_y = all_coords.max(axis=0) + self.domain_pad
        
        n_x = int((max_x - min_x) / self.mesh_max_edge) + 1
        n_y = int((max_y - min_y) / self.mesh_max_edge) + 1
        
        x_lin = np.linspace(min_x, max_x, n_x)
        y_lin = np.linspace(min_y, max_y, n_y)
        xx, yy = np.meshgrid(x_lin, y_lin)
        mesh_nodes = np.column_stack([xx.ravel(), yy.ravel()])
        
        self.tri_ = Delaunay(mesh_nodes)
        self.coords_ = mesh_nodes
        
        if self.verbose: print("[Mesh] Assembling SPDE matrices...")
        self.Q_base_, C_mass = spde_precision_with_mass(
            self.coords_, self.tri_.simplices, kappa=self.kappa
        )
        self.w_ = np.array(C_mass.sum(axis=1)).flatten()
        
        if self.verbose: print(f"[Mesh] Created {self.coords_.shape[0]} nodes.")

    def _get_gene_coords(self, gene_name):
        subset = self.df_[self.df_['feature_name'] == gene_name]
        if len(subset) == 0:
            raise ValueError(f"Gene '{gene_name}' not found.")
        return subset[['x_location', 'y_location']].values

    def _compute_covariate_field(self, gene_name):
        if self.verbose: print(f"[Covariate] project '{gene_name}' to mesh...")
        coords = self._get_gene_coords(gene_name)
        A_cov = barycentric_A(coords, self.tri_)
        counts_per_node = np.array(A_cov.sum(axis=0)).flatten()
        intensity_per_node = counts_per_node / (self.w_ + 1e-9)
        
        mu = intensity_per_node.mean()
        sd = intensity_per_node.std() + 1e-9
        return (intensity_per_node - mu) / sd

    def fit(self, target_gene, covariate_genes=None):
        if self.tri_ is None:
            self._build_mesh()
            
        target_coords = self._get_gene_coords(target_gene)
        N_obs = target_coords.shape[0]
        if self.verbose: print(f"[Fit] Target: {target_gene} ({N_obs} events)")
        
        A = barycentric_A(target_coords, self.tri_)
        A_tilde = csr_matrix(np.eye(self.coords_.shape[0]))
        
        X_cols_node = [np.ones((self.coords_.shape[0]))]
        X_cols_event = [np.ones(N_obs)]
        
        # Store as a list to handle multiple covariates
        self.covariate_names_ = []
        if covariate_genes:
            # Standardize input to a list if they passed a single string
            if isinstance(covariate_genes, str):
                covariate_genes = [covariate_genes]
            self.covariate_names_ = covariate_genes
            
            # Loop through all provided covariates
            for cov_gene in covariate_genes:
                cov_field_nodes = self._compute_covariate_field(cov_gene)
                cov_vals_events = A @ cov_field_nodes                
                X_cols_node.append(cov_field_nodes)
                X_cols_event.append(cov_vals_events)
            
        X_tilde = np.column_stack(X_cols_node)
        X       = np.column_stack(X_cols_event)
        
        p = X.shape[1]
        beta_init = np.zeros(p)
        beta_init[0] = np.log(N_obs / np.sum(self.w_) + 1e-9)
        
        # Pass the dynamically sized X_tilde and X to the engine
        self.model_ = VoGCAM_SPDE_GPU(verbose=self.verbose, rank=self.rank, max_iter=100)
        self.model_.fit(A_tilde, A, X_tilde, X, self.w_, self.Q_base_, beta_init)
        return self

    def plot_results(self):
        """
        Plots:
        1. Fitted Intensity (Lambda)
        2. Posterior Mean of Latent Field (Mu)
        3+. Covariate Fields (if present)
        """
        if self.model_ is None: 
            print("Model not fitted.")
            return
        
        A_id = csr_matrix(np.eye(self.coords_.shape[0]))
        X_cols = [np.ones((self.coords_.shape[0]))]
        
        # Rebuild the covariate fields for plotting
        if hasattr(self, 'covariate_names_') and self.covariate_names_:
            for cov_name in self.covariate_names_:
                cov = self._compute_covariate_field(cov_name)
                X_cols.append(cov)
        
        X_pred = np.column_stack(X_cols)
        lam = self.model_.expected_intensity_on_grid(A_id, X_pred).get()
        mu_post = self.model_.mu_.get()

        # Determine Layout based on number of covariates
        n_covs = len(self.covariate_names_) if hasattr(self, 'covariate_names_') else 0
        n_plots = 2 + n_covs
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1: axes = [axes] # Safety fallback
        
        # --- PLOT A: Fitted Intensity ---
        ax = axes[0]
        cnt = ax.tricontourf(self.coords_[:,0], self.coords_[:,1], 
                             self.tri_.simplices, lam, levels=14, cmap='viridis') 
        plt.colorbar(cnt, ax=ax, label="Intensity $\lambda(s)$")
        ax.set_title(f"Fitted Intensity\n(Baseline $\\beta_0$: {self.model_.beta_[0].item():.2f})")
        
        # --- PLOT B: Posterior Mean ---
        ax = axes[1]
        vmin, vmax = mu_post.min(), mu_post.max()
        if vmin >= 0: vmin = -0.1
        if vmax <= 0: vmax = 0.1
            
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        cnt = ax.tricontourf(self.coords_[:,0], self.coords_[:,1], 
                             self.tri_.simplices, mu_post, 
                             levels=14, cmap='coolwarm', norm=norm)
        plt.colorbar(cnt, ax=ax, label="Latent Field $\mu(s)$")
        ax.set_title("Posterior Mean (Spatial Residual)")

        # --- PLOT C+: Covariates ---
        for i, cov_name in enumerate(self.covariate_names_):
            ax = axes[2 + i]
            cov_field = X_cols[1 + i]
            vmax_cov = np.max(np.abs(cov_field))
            
            cnt = ax.tricontourf(self.coords_[:,0], self.coords_[:,1], 
                                 self.tri_.simplices, cov_field, 
                                 levels=14, cmap='coolwarm', vmin=-vmax_cov, vmax=vmax_cov)
            plt.colorbar(cnt, ax=ax, label="Std. Intensity (Z-Score)")
            
            # Extract the specific beta coefficient for this covariate
            beta_cov = self.model_.beta_[1 + i].item()
            ax.set_title(f"Covariate: {cov_name}\n$\\beta_{i+1}$ = {beta_cov:.3f}")

        plt.tight_layout()
        plt.show()
    
    def plot_loss(self):
        """
        Plots the ELBO (Evidence Lower Bound) history to check convergence.
        """
        if self.model_ is None or not hasattr(self.model_, 'elbo_history_'):
            print("Model not fitted yet.")
            return

        elbos = [float(e) for e in self.model_.elbo_history_]
        
        plt.figure(figsize=(6, 4))
        plt.plot(elbos, 'b-', linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO (Higher is better)")
        plt.title("Optimization Convergence")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()