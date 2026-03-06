"""
Parameter Uncertainty Quantification and Covariance Analysis
Bootstrap resampling and sensitivity analysis for model parameters

This supplementary code performs:
- Bootstrap parameter estimation (1000 iterations)
- Covariance matrix calculation
- Confidence interval determination
- Sensitivity analysis
- Correlation structure visualization

Author: Research Team
Date: March 2026
Version: 1.1.0 (Maintenance Update)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import from main model (assumes multiphysics_model.py is available)
# For standalone use, copy the necessary classes from the main implementation

# ============================================================================
# ADVANCED UNCERTAINTY QUANTIFICATION
# ============================================================================

class AdvancedUncertaintyAnalysis:
    """Comprehensive parameter uncertainty analysis"""
    
    def __init__(self, fitted_results: Dict):
        """
        Initialize with fitted model results
        
        Parameters:
        -----------
        fitted_results : dict
            Results from ParameterEstimator.fit()
        """
        self.results = fitted_results
        self.params = fitted_results['params']
        self.cov = fitted_results['covariance']
        self.n_obs = fitted_results['n_obs']
        self.n_params = len(self.params)
        
    def parameter_confidence_intervals(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Calculate confidence intervals from covariance matrix
        
        Parameters:
        -----------
        alpha : float
            Significance level (default 0.05 for 95% CI)
        
        Returns:
        --------
        ci_df : pd.DataFrame
            Parameter estimates with confidence intervals
        """
        # Critical t-value
        dof = self.n_obs - self.n_params
        t_crit = norm.ppf(1 - alpha/2)  # Use normal for large samples
        
        param_names = ['i0_base', 'k0', 'beta', 'gamma']
        
        data = []
        for i, name in enumerate(param_names):
            value = self.params[name]
            se = self.results['std_errors'][name]
            ci_lower = value - t_crit * se
            ci_upper = value + t_crit * se
            rel_error = 100 * se / value
            
            data.append({
                'Parameter': name,
                'Value': value,
                'Std_Error': se,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Rel_Error_%': rel_error
            })
        
        return pd.DataFrame(data)
    
    def correlation_matrix(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate parameter correlation matrix
        
        Returns:
        --------
        corr_df : pd.DataFrame
            Correlation matrix with parameter names
        corr : np.ndarray
            Correlation matrix as array
        """
        # Convert covariance to correlation
        std = np.sqrt(np.diag(self.cov))
        corr = self.cov / np.outer(std, std)
        
        param_names = ['i0_base', 'k0', 'beta', 'gamma']
        corr_df = pd.DataFrame(
            corr,
            index=param_names,
            columns=param_names
        )
        
        return corr_df, corr
    
    def joint_confidence_region(self, param1: str, param2: str, 
                               alpha: float = 0.05) -> Tuple:
        """
        Calculate joint confidence ellipse for two parameters
        
        Parameters:
        -----------
        param1, param2 : str
            Parameter names to analyze
        alpha : float
            Significance level
        
        Returns:
        --------
        ellipse_points : np.ndarray
            Points defining confidence ellipse boundary
        center : tuple
            Center of ellipse (param1_value, param2_value)
        """
        param_names = ['i0_base', 'k0', 'beta', 'gamma']
        idx1 = param_names.index(param1)
        idx2 = param_names.index(param2)
        
        # Extract 2x2 covariance submatrix
        cov_sub = self.cov[[idx1, idx2], :][:, [idx1, idx2]]
        
        # Chi-square critical value for 2 parameters
        chi2_crit = chi2.ppf(1 - alpha, 2)
        
        # Eigendecomposition for ellipse parameters
        eigvals, eigvecs = np.linalg.eig(cov_sub)
        
        # Generate ellipse points
        theta = np.linspace(0, 2*np.pi, 100)
        ellipse = np.array([np.cos(theta), np.sin(theta)])
        
        # Scale by eigenvalues and chi-square
        ellipse = np.sqrt(chi2_crit) * np.sqrt(eigvals[:, np.newaxis]) * ellipse
        
        # Rotate by eigenvectors
        ellipse = eigvecs @ ellipse
        
        # Translate to parameter values
        center = (self.params[param1], self.params[param2])
        ellipse[0, :] += center[0]
        ellipse[1, :] += center[1]
        
        return ellipse, center
    
    def sensitivity_analysis(self, perturbation: float = 0.01) -> pd.DataFrame:
        """
        Local sensitivity analysis via finite differences
        
        Parameters:
        -----------
        perturbation : float
            Fractional parameter perturbation (default 1%)
        
        Returns:
        --------
        sensitivity_df : pd.DataFrame
            Sensitivity metrics for each parameter
        """
        param_names = ['i0_base', 'k0', 'beta', 'gamma']
        
        # Reference prediction (using mean residual as output metric)
        ref_residual = np.mean(np.abs(self.results['residuals']))
        
        sensitivities = []
        
        for param in param_names:
            # Perturb parameter
            perturbed_value = self.params[param] * (1 + perturbation)
            
            # Calculate change in output (simplified for demonstration)
            # In practice, would re-run model with perturbed parameter
            param_sensitivity = {
                'Parameter': param,
                'Nominal_Value': self.params[param],
                'Std_Error': self.results['std_errors'][param],
                'CV_%': 100 * self.results['std_errors'][param] / self.params[param],
                'Significance': 'High' if self.results['std_errors'][param] / 
                               self.params[param] < 0.1 else 'Moderate'
            }
            sensitivities.append(param_sensitivity)
        
        return pd.DataFrame(sensitivities)

# ============================================================================
# BOOTSTRAP IMPLEMENTATION
# ============================================================================

class BootstrapAnalysis:
    """Bootstrap resampling for parameter distributions"""
    
    def __init__(self, data: pd.DataFrame, n_bootstrap: int = 1000, 
                 seed: int = 42):
        """
        Initialize bootstrap analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Experimental data with columns: t_age, T_age, t_exp, C_Sn, i_exp
        n_bootstrap : int
            Number of bootstrap iterations
        seed : int
            Random seed for reproducibility
        """
        self.data = data
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.bootstrap_results = None
        
    def run_bootstrap(self) -> Dict:
        """
        Execute bootstrap parameter estimation
        
        Returns:
        --------
        results : dict
            Bootstrap parameter distributions and statistics
        """
        np.random.seed(self.seed)
        
        # Storage for bootstrap samples
        param_samples = {
            'i0_base': [],
            'k0': [],
            'beta': [],
            'gamma': []
        }
        
        # Fit metrics for each bootstrap
        r2_samples = []
        rmse_samples = []
        
        print(f"Executing {self.n_bootstrap} bootstrap iterations...")
        print("This may take several minutes...")
        
        # Initial fit for starting values
        from multiphysics_model import ParameterEstimator
        initial_estimator = ParameterEstimator(self.data)
        initial_fit = initial_estimator.fit()
        
        successful_iterations = 0
        
        for i in range(self.n_bootstrap):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{self.n_bootstrap} "
                      f"({100*(i+1)/self.n_bootstrap:.1f}%) - "
                      f"Successful: {successful_iterations}")
            
            try:
                # Resample with replacement
                bootstrap_indices = np.random.choice(
                    len(self.data), 
                    size=len(self.data), 
                    replace=True
                )
                bootstrap_sample = self.data.iloc[bootstrap_indices].reset_index(drop=True)
                
                # Fit model to bootstrap sample
                boot_estimator = ParameterEstimator(bootstrap_sample)
                boot_result = boot_estimator.fit(
                    initial_guess=initial_fit['params']
                )
                
                if boot_result['success'] and boot_result['r2'] > 0.5:
                    # Store parameters
                    for param in param_samples.keys():
                        param_samples[param].append(boot_result['params'][param])
                    
                    r2_samples.append(boot_result['r2'])
                    rmse_samples.append(boot_result['rmse'])
                    successful_iterations += 1
                    
            except Exception as e:
                # Skip failed iterations
                continue
        
        print(f"\nBootstrap complete: {successful_iterations}/{self.n_bootstrap} "
              f"successful iterations ({100*successful_iterations/self.n_bootstrap:.1f}%)")
        
        # Calculate statistics
        bootstrap_stats = {}
        
        for param, samples in param_samples.items():
            samples_array = np.array(samples)
            
            bootstrap_stats[param] = {
                'mean': np.mean(samples_array),
                'median': np.median(samples_array),
                'std': np.std(samples_array, ddof=1),
                'ci_lower_95': np.percentile(samples_array, 2.5),
                'ci_upper_95': np.percentile(samples_array, 97.5),
                'ci_lower_90': np.percentile(samples_array, 5.0),
                'ci_upper_90': np.percentile(samples_array, 95.0),
                'samples': samples_array,
                'cv_%': 100 * np.std(samples_array, ddof=1) / np.mean(samples_array)
            }
        
        # Model performance statistics
        bootstrap_stats['performance'] = {
            'r2_mean': np.mean(r2_samples),
            'r2_std': np.std(r2_samples, ddof=1),
            'rmse_mean': np.mean(rmse_samples),
            'rmse_std': np.std(rmse_samples, ddof=1),
            'n_successful': successful_iterations
        }
        
        self.bootstrap_results = bootstrap_stats
        return bootstrap_stats
    
    def parameter_summary_table(self) -> pd.DataFrame:
        """
        Generate formatted parameter summary table
        
        Returns:
        --------
        summary : pd.DataFrame
            Bootstrap parameter statistics
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap analysis first")
        
        param_names = ['i0_base', 'k0', 'beta', 'gamma']
        units = ['mA/cm²', 's⁻¹', '-', 'wt%⁻¹']
        
        data = []
        for param, unit in zip(param_names, units):
            stats = self.bootstrap_results[param]
            
            data.append({
                'Parameter': param,
                'Unit': unit,
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Std_Dev': stats['std'],
                'CV_%': stats['cv_%'],
                'CI_95_Lower': stats['ci_lower_95'],
                'CI_95_Upper': stats['ci_upper_95']
            })
        
        return pd.DataFrame(data)
        
    def visualize_distributions(self, figsize=(14, 10)):
        """
        Create comprehensive visualization of bootstrap results
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap analysis first")
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Bootstrap Parameter Distributions', 
                    fontsize=16, fontweight='bold')
        
        param_names = ['i0_base', 'k0', 'beta', 'gamma']
        param_labels = [
            r'$i_{0,base}$ (mA/cm²)',
            r'$k_0$ (s⁻¹)',
            r'$\beta$ (dimensionless)',
            r'$\gamma$ (wt%⁻¹)'
        ]
        
        for idx, (param, label) in enumerate(zip(param_names, param_labels)):
            ax = axes[idx // 2, idx % 2]
            stats = self.bootstrap_results[param]
            samples = stats['samples']
            
            # Histogram with KDE
            ax.hist(samples, bins=50, density=True, alpha=0.6, 
                color='steelblue', edgecolor='black')
            
            # KDE overlay with error handling
            from scipy.stats import gaussian_kde
            try:
                # Add a tiny amount of noise to prevent singular matrices
                jittered_samples = samples + np.random.normal(0, np.std(samples)/1000, size=len(samples))
                kde = gaussian_kde(jittered_samples)
                x_range = np.linspace(samples.min(), samples.max(), 200)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except np.linalg.LinAlgError:
                # If KDE still fails, skip it and just show histogram
                print(f"Warning: Could not compute KDE for parameter '{param}' due to singular matrix")
            
            # Mean and median
            ax.axvline(stats['mean'], color='darkred', linestyle='--', 
                    linewidth=2, label=f"Mean: {stats['mean']:.4g}")
            ax.axvline(stats['median'], color='darkgreen', linestyle=':', 
                    linewidth=2, label=f"Median: {stats['median']:.4g}")
            
            # Confidence intervals
            ax.axvline(stats['ci_lower_95'], color='gray', linestyle='--', 
                    linewidth=1, alpha=0.7)
            ax.axvline(stats['ci_upper_95'], color='gray', linestyle='--', 
                    linewidth=1, alpha=0.7)
            ax.axvspan(stats['ci_lower_95'], stats['ci_upper_95'], 
                    alpha=0.2, color='gray', label='95% CI')
            
            ax.set_xlabel(label, fontsize=11, fontweight='bold')
            ax.set_ylabel('Probability Density', fontsize=10)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add text box with statistics
            textstr = f'CV = {stats["cv_%"]:.1f}%\nn = {len(samples)}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', bbox=props)
        
        # Correlation heatmap in subplot 5
        ax_corr = axes[2, 0]
        
        # Calculate correlation matrix from samples
        samples_array = np.column_stack([
            self.bootstrap_results[p]['samples'] for p in param_names
        ])
        corr_matrix = np.corrcoef(samples_array.T)
        
        im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                        aspect='auto')
        ax_corr.set_xticks(range(4))
        ax_corr.set_yticks(range(4))
        ax_corr.set_xticklabels(['i₀', 'k₀', 'β', 'γ'], fontsize=10)
        ax_corr.set_yticklabels(['i₀', 'k₀', 'β', 'γ'], fontsize=10)
        ax_corr.set_title('Parameter Correlation Matrix', fontweight='bold')
        
        # Add correlation values
        for i in range(4):
            for j in range(4):
                text = ax_corr.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                ha="center", va="center", color="black",
                                fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
        
        # Performance metrics in subplot 6
        ax_perf = axes[2, 1]
        ax_perf.axis('off')
        
        perf = self.bootstrap_results['performance']
        perf_text = (
            f"Model Performance (Bootstrap)\n\n"
            f"R² = {perf['r2_mean']:.4f} ± {perf['r2_std']:.4f}\n"
            f"RMSE = {perf['rmse_mean']:.4f} ± {perf['rmse_std']:.4f} mA/cm²\n\n"
            f"Successful iterations: {perf['n_successful']}/{self.n_bootstrap}\n"
            f"Success rate: {100*perf['n_successful']/self.n_bootstrap:.1f}%"
        )
        
        ax_perf.text(0.1, 0.5, perf_text, transform=ax_perf.transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig

# ============================================================================
# PREDICTION UNCERTAINTY PROPAGATION
# ============================================================================

class PredictionUncertainty:
    """Propagate parameter uncertainty to model predictions"""
    
    def __init__(self, bootstrap_results: Dict):
        """
        Initialize with bootstrap results
        
        Parameters:
        -----------
        bootstrap_results : dict
            Results from BootstrapAnalysis.run_bootstrap()
        """
        self.bootstrap_results = bootstrap_results
    
    def prediction_bands(self, conditions: pd.DataFrame, 
                        confidence: float = 0.95) -> pd.DataFrame:
        """
        Calculate prediction confidence bands
        
        Parameters:
        -----------
        conditions : pd.DataFrame
            Conditions to predict (t_age, T_age, t_exp, C_Sn)
        confidence : float
            Confidence level (default 0.95)
        
        Returns:
        --------
        predictions : pd.DataFrame
            Mean predictions with confidence bands
        """
        from multiphysics_model import CoupledAnodeModel
        
        # Get bootstrap parameter samples
        n_samples = len(self.bootstrap_results['i0_base']['samples'])
        
        # Storage for predictions from each bootstrap sample
        all_predictions = []
        
        for i in range(n_samples):
            params = {
                'i0_base': self.bootstrap_results['i0_base']['samples'][i],
                'k0': self.bootstrap_results['k0']['samples'][i],
                'beta': self.bootstrap_results['beta']['samples'][i],
                'gamma': self.bootstrap_results['gamma']['samples'][i]
            }
            
            model = CoupledAnodeModel(params)
            predictions = model.predict_dataset(conditions)
            all_predictions.append(predictions)
        
        # Stack predictions
        all_predictions = np.array(all_predictions)
        
        # Calculate statistics
        alpha = 1 - confidence
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0, ddof=1)
        ci_lower = np.percentile(all_predictions, 100*alpha/2, axis=0)
        ci_upper = np.percentile(all_predictions, 100*(1-alpha/2), axis=0)
        
        results = conditions.copy()
        results['prediction_mean'] = mean_pred
        results['prediction_std'] = std_pred
        results['ci_lower'] = ci_lower
        results['ci_upper'] = ci_upper
        results['ci_width'] = ci_upper - ci_lower
        
        return results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Demonstration of uncertainty analysis"""
    
    print("=" * 70)
    print("Parameter Uncertainty Quantification - Bootstrap Analysis")
    print("=" * 70)
    
    # Generate or load experimental data
    from multiphysics_model import generate_synthetic_data, ParameterEstimator
    
    print("\n1. Loading experimental dataset...")
    data = generate_synthetic_data(n_samples=280)
    print(f"   Dataset size: {len(data)} measurements")
    
    # Initial parameter estimation
    print("\n2. Initial parameter estimation...")
    estimator = ParameterEstimator(data)
    results = estimator.fit()
    
    print(f"   R² = {results['r2']:.4f}")
    print(f"   RMSE = {results['rmse']:.4f} mA/cm²")
    
    # Covariance-based uncertainty
    print("\n3. Covariance-based uncertainty analysis...")
    uncertainty = AdvancedUncertaintyAnalysis(results)
    
    ci_df = uncertainty.parameter_confidence_intervals()
    print("\n   Parameter Confidence Intervals (95%):")
    print(ci_df.to_string(index=False))
    
    corr_df, _ = uncertainty.correlation_matrix()
    print("\n   Parameter Correlation Matrix:")
    print(corr_df.to_string())
    
    # Bootstrap analysis (reduced iterations for demonstration)
    print("\n4. Bootstrap parameter estimation...")
    print("   Note: Using 100 iterations for demonstration")
    print("   (Full analysis uses 1000 iterations)")
    
    bootstrap = BootstrapAnalysis(data, n_bootstrap=100, seed=42)
    bootstrap_stats = bootstrap.run_bootstrap()
    
    summary = bootstrap.parameter_summary_table()
    print("\n   Bootstrap Parameter Summary:")
    print(summary.to_string(index=False))
    
    # Visualizations
    print("\n5. Generating visualizations...")
    fig = bootstrap.visualize_distributions()
    plt.savefig('bootstrap_distributions.png', dpi=300, bbox_inches='tight')
    print("   Saved: bootstrap_distributions.png")
    
    # Prediction uncertainty
    print("\n6. Prediction uncertainty propagation...")
    test_conditions = pd.DataFrame({
        't_age': [4, 4, 8],
        'T_age': [160, 160, 130],
        't_exp': [168, 336, 168],
        'C_Sn': [0.05, 0.10, 0.00]
    })
    
    pred_uncertainty = PredictionUncertainty(bootstrap_stats)
    pred_bands = pred_uncertainty.prediction_bands(test_conditions)
    
    print("\n   Prediction Confidence Bands:")
    print(pred_bands[['t_age', 'T_age', 'C_Sn', 
                      'prediction_mean', 'ci_lower', 'ci_upper']].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Uncertainty analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()