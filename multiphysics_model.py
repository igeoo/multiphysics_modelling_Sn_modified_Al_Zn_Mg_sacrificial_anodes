"""
Multi-Physics Modelling of Sn-Modified Al-Zn-Mg Sacrificial Anodes
Complete Implementation Code

This module implements the coupled precipitation-electrochemistry model
described in the manuscript, including:
- JMAK precipitation kinetics
- Butler-Volmer electrochemical kinetics
- Microstructure-electrochemistry coupling
- Parameter estimation and uncertainty quantification
- Service life prediction

Author: Research Team
Date: March 2026
Version: 1.1.0 (Maintenance Update)
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import odeint
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

@dataclass
class PhysicalConstants:
    """Physical constants for electrochemical calculations"""
    R: float = 8.314  # Gas constant [J/(mol·K)]
    F: float = 96485  # Faraday constant [C/mol]
    T: float = 298.15  # Temperature [K] (25°C)
    n: float = 3.0  # Electron transfer number for Al → Al³⁺
    E0_Al: float = -1.662  # Standard potential Al/Al³⁺ vs SHE [V]
    M_Al: float = 26.98  # Molar mass of aluminum [g/mol]
    rho_Al: float = 2.70  # Density of aluminum [g/cm³]
    alpha_a: float = 0.5  # Anodic transfer coefficient
    
    # JMAK parameters (fixed from literature)
    Q: float = 125000  # Activation energy [J/mol]
    n_avrami: float = 2.0  # Avrami exponent
    m_power: float = 0.68  # Power-law exponent for precipitation coupling
    
    # Anode efficiency
    eta_anode: float = 0.90  # Current efficiency (90%)

CONST = PhysicalConstants()

# ============================================================================
# PRECIPITATION KINETICS MODEL
# ============================================================================

class JMAKModel:
    """Johnson-Mehl-Avrami-Kolmogorov precipitation kinetics"""
    
    def __init__(self, k0: float, Q: float = CONST.Q, n: float = CONST.n_avrami):
        """
        Initialize JMAK model
        
        Parameters:
        -----------
        k0 : float
            Pre-exponential rate constant [s⁻¹]
        Q : float
            Activation energy [J/mol]
        n : float
            Avrami exponent (dimensionless)
        """
        self.k0 = k0
        self.Q = Q
        self.n = n
    
    def rate_constant(self, T: float) -> float:
        """
        Calculate temperature-dependent rate constant
        
        Parameters:
        -----------
        T : float
            Temperature [K]
        
        Returns:
        --------
        k : float
            Rate constant [s⁻¹]
        """
        return self.k0 * np.exp(-self.Q / (CONST.R * T))
    
    def volume_fraction(self, t: float, T: float) -> float:
        """
        Calculate transformed volume fraction
        
        Parameters:
        -----------
        t : float
            Time [s]
        T : float
            Temperature [K]
        
        Returns:
        --------
        X : float
            Volume fraction transformed (0 to 1)
        """
        k = self.rate_constant(T)
        return 1 - np.exp(-(k * t) ** self.n)
    
    def characteristic_time(self, T: float, fraction: float = 0.5) -> float:
        """
        Calculate characteristic transformation time
        
        Parameters:
        -----------
        T : float
            Temperature [K]
        fraction : float
            Target volume fraction (default 0.5)
        
        Returns:
        --------
        tau : float
            Characteristic time [hours]
        """
        k = self.rate_constant(T)
        tau_seconds = ((-np.log(1 - fraction)) ** (1/self.n)) / k
        return tau_seconds / 3600  # Convert to hours

# ============================================================================
# ELECTROCHEMICAL MODEL
# ============================================================================

class ElectrochemicalModel:
    """Butler-Volmer electrochemical kinetics with precipitation coupling"""
    
    def __init__(self, i0_base: float, beta: float, gamma: float, 
                 m: float = CONST.m_power):
        """
        Initialize electrochemical model
        
        Parameters:
        -----------
        i0_base : float
            Baseline exchange current density [mA/cm²]
        beta : float
            Precipitation coupling coefficient (dimensionless)
        gamma : float
            Compositional sensitivity [wt%⁻¹]
        m : float
            Power-law exponent for precipitation effect
        """
        self.i0_base = i0_base
        self.beta = beta
        self.gamma = gamma
        self.m = m
    
    def exchange_current(self, X: float, C_Sn: float) -> float:
        """
        Calculate exchange current density with coupling
        
        Parameters:
        -----------
        X : float
            Precipitate volume fraction (0 to 1)
        C_Sn : float
            Sn composition [wt%]
        
        Returns:
        --------
        i0 : float
            Exchange current density [mA/cm²]
        """
        precipitation_term = 1 + self.beta * (X ** self.m)
        composition_term = 1 + self.gamma * C_Sn
        return self.i0_base * precipitation_term * composition_term
    
    def nernst_potential(self, a_Al3: float = 1e-6) -> float:
        """
        Calculate equilibrium potential via Nernst equation
        
        Parameters:
        -----------
        a_Al3 : float
            Activity of Al³⁺ ions (dimensionless)
        
        Returns:
        --------
        E_eq : float
            Equilibrium potential [V vs SHE]
        """
        return CONST.E0_Al - (CONST.R * CONST.T / (CONST.n * CONST.F)) * \
               np.log(a_Al3)
    
    def current_density(self, eta: float, i0: float) -> float:
        """
        Calculate current density via Butler-Volmer equation
        (Tafel approximation for large anodic overpotential)
        
        Parameters:
        -----------
        eta : float
            Overpotential [V]
        i0 : float
            Exchange current density [mA/cm²]
        
        Returns:
        --------
        i : float
            Current density [mA/cm²]
        """
        exponent = (CONST.alpha_a * CONST.n * CONST.F * eta) / (CONST.R * CONST.T)
        return i0 * np.exp(exponent)

# ============================================================================
# COUPLED MULTI-PHYSICS MODEL
# ============================================================================

class CoupledAnodeModel:
    """Integrated precipitation-electrochemistry model"""
    
    def __init__(self, params: Dict[str, float]):
        """
        Initialize coupled model
        
        Parameters:
        -----------
        params : dict
            Dictionary containing model parameters:
            - i0_base: baseline exchange current [mA/cm²]
            - k0: precipitation rate constant [s⁻¹]
            - beta: precipitation coupling coefficient
            - gamma: compositional sensitivity [wt%⁻¹]
        """
        self.jmak = JMAKModel(k0=params['k0'])
        self.electrochem = ElectrochemicalModel(
            i0_base=params['i0_base'],
            beta=params['beta'],
            gamma=params['gamma']
        )
        self.params = params
    
    def predict_current(self, t_age: float, T_age: float, 
                       t_exp: float, C_Sn: float) -> float:
        """
        Predict current density for given conditions
        
        Parameters:
        -----------
        t_age : float
            Aging time [hours]
        T_age : float
            Aging temperature [°C]
        t_exp : float
            Exposure time [hours]
        C_Sn : float
            Sn composition [wt%]
        
        Returns:
        --------
        i_avg : float
            Time-averaged current density [mA/cm²]
        """
        # Convert units
        t_age_s = t_age * 3600  # hours to seconds
        T_age_K = T_age + 273.15  # Celsius to Kelvin
        
        # Calculate precipitation state
        X = self.jmak.volume_fraction(t_age_s, T_age_K)
        
        # Calculate exchange current density
        i0 = self.electrochem.exchange_current(X, C_Sn)
        
        # Simplified passivation decay (phenomenological)
        # This represents corrosion product accumulation
        decay_factor = 1 / (1 + 0.0015 * t_exp)
        
        # Typical overpotential for sacrificial anode operation
        eta = 0.15  # ~150 mV anodic overpotential
        
        # Calculate instantaneous current
        i_inst = self.electrochem.current_density(eta, i0)
        
        # Apply decay and efficiency
        i_avg = i_inst * decay_factor * CONST.eta_anode
        
        return i_avg
    
    def predict_dataset(self, conditions: pd.DataFrame) -> np.ndarray:
        """
        Predict current densities for multiple conditions
        
        Parameters:
        -----------
        conditions : pd.DataFrame
            DataFrame with columns: t_age, T_age, t_exp, C_Sn
        
        Returns:
        --------
        predictions : np.ndarray
            Array of predicted current densities [mA/cm²]
        """
        predictions = np.array([
            self.predict_current(
                row['t_age'], row['T_age'], 
                row['t_exp'], row['C_Sn']
            )
            for _, row in conditions.iterrows()
        ])
        return predictions

# ============================================================================
# PARAMETER ESTIMATION
# ============================================================================

class ParameterEstimator:
    """Parameter estimation via nonlinear least squares"""
    
    def __init__(self, experimental_data: pd.DataFrame):
        """
        Initialize estimator with experimental data
        
        Parameters:
        -----------
        experimental_data : pd.DataFrame
            Must contain columns: t_age, T_age, t_exp, C_Sn, i_exp
        """
        self.data = experimental_data
        self.n_obs = len(experimental_data)
    
    def objective_function(self, params_array: np.ndarray) -> np.ndarray:
        """
        Weighted least squares objective
        
        Parameters:
        -----------
        params_array : np.ndarray
            Array of parameters [i0_base, k0, beta, gamma]
        
        Returns:
        --------
        residuals : np.ndarray
            Weighted residuals
        """
        params = {
            'i0_base': params_array[0],
            'k0': params_array[1],
            'beta': params_array[2],
            'gamma': params_array[3]
        }
        
        model = CoupledAnodeModel(params)
        predictions = model.predict_dataset(self.data)
        
        # Weighted residuals (weight by inverse of measurement magnitude)
        weights = 1 / (self.data['i_exp'].values + 0.1)
        residuals = weights * (self.data['i_exp'].values - predictions)
        
        return residuals
    
    def fit(self, initial_guess: Dict[str, float] = None) -> Dict:
        """
        Fit model parameters to experimental data
        
        Parameters:
        -----------
        initial_guess : dict, optional
            Initial parameter values
        
        Returns:
        --------
        results : dict
            Fitted parameters and statistics
        """
        if initial_guess is None:
            initial_guess = {
                'i0_base': 1.5,
                'k0': 5e-5,
                'beta': 2.0,
                'gamma': 8.0
            }
        
        # Parameter bounds (physically motivated)
        bounds = (
            [0.1, 1e-7, 0.0, -5.0],  # Lower bounds
            [10.0, 1e-3, 5.0, 15.0]  # Upper bounds
        )
        
        x0 = np.array([initial_guess[k] for k in 
                      ['i0_base', 'k0', 'beta', 'gamma']])
        
        # Levenberg-Marquardt optimization
        result = least_squares(
            self.objective_function,
            x0,
            bounds=bounds,
            method='trf',  # Trust Region Reflective
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=500,
            verbose=0
        )
        
        # Extract fitted parameters
        fitted_params = {
            'i0_base': result.x[0],
            'k0': result.x[1],
            'beta': result.x[2],
            'gamma': result.x[3]
        }
        
        # Calculate predictions and residuals
        model = CoupledAnodeModel(fitted_params)
        predictions = model.predict_dataset(self.data)
        residuals = self.data['i_exp'].values - predictions
        
        # Performance metrics
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.data['i_exp'].values - 
                        self.data['i_exp'].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(ss_res / self.n_obs)
        mape = 100 * np.mean(np.abs(residuals / self.data['i_exp'].values))
        
        # Estimate parameter uncertainties from Jacobian
        J = result.jac
        try:
            # Covariance matrix estimation
            cov = np.linalg.inv(J.T @ J) * (ss_res / (self.n_obs - 4))
            std_errors = np.sqrt(np.diag(cov))
        except:
            std_errors = np.zeros(4)
            cov = np.eye(4)
        
        return {
            'params': fitted_params,
            'std_errors': {
                'i0_base': std_errors[0],
                'k0': std_errors[1],
                'beta': std_errors[2],
                'gamma': std_errors[3]
            },
            'covariance': cov,
            'r2': r2,
            'rmse': rmse,
            'mape': mape,
            'n_obs': self.n_obs,
            'n_iter': result.nfev,
            'predictions': predictions,
            'residuals': residuals,
            'success': result.success
        }

# ============================================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================================

class UncertaintyAnalysis:
    """Bootstrap resampling for parameter uncertainty"""
    
    def __init__(self, data: pd.DataFrame, n_bootstrap: int = 1000):
        """
        Initialize uncertainty analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Experimental data
        n_bootstrap : int
            Number of bootstrap samples
        """
        self.data = data
        self.n_bootstrap = n_bootstrap
    
    def bootstrap_fit(self, seed: int = 42) -> Dict:
        """
        Perform bootstrap parameter estimation
        
        Parameters:
        -----------
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        results : dict
            Bootstrap parameter distributions
        """
        np.random.seed(seed)
        
        param_distributions = {
            'i0_base': [],
            'k0': [],
            'beta': [],
            'gamma': []
        }
        
        # Initial fit for starting values
        estimator = ParameterEstimator(self.data)
        initial_fit = estimator.fit()
        
        print(f"Running {self.n_bootstrap} bootstrap iterations...")
        
        for i in range(self.n_bootstrap):
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{self.n_bootstrap}")
            
            # Resample with replacement
            bootstrap_sample = self.data.sample(
                n=len(self.data), 
                replace=True
            )
            
            # Fit to bootstrap sample
            try:
                boot_estimator = ParameterEstimator(bootstrap_sample)
                boot_result = boot_estimator.fit(
                    initial_guess=initial_fit['params']
                )
                
                if boot_result['success']:
                    for key in param_distributions.keys():
                        param_distributions[key].append(
                            boot_result['params'][key]
                        )
            except:
                continue
        
        # Calculate statistics
        bootstrap_stats = {}
        for param, values in param_distributions.items():
            values = np.array(values)
            bootstrap_stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'samples': values
            }
        
        return bootstrap_stats

# ============================================================================
# SERVICE LIFE PREDICTION
# ============================================================================

class ServiceLifeModel:
    """Long-term service life prediction"""
    
    def __init__(self, model: CoupledAnodeModel, 
                 r0: float = 2.0, L: float = 10.0):
        """
        Initialize service life model
        
        Parameters:
        -----------
        model : CoupledAnodeModel
            Fitted electrochemical model
        r0 : float
            Initial anode radius [cm]
        L : float
            Anode length [cm]
        """
        self.model = model
        self.r0 = r0
        self.L = L
    
    def integrate_mass_loss(self, t_age: float, T_age: float, 
                           C_Sn: float, t_service: float) -> Tuple:
        """
        Integrate current density to predict mass loss
        
        Parameters:
        -----------
        t_age : float
            Aging time [hours]
        T_age : float
            Aging temperature [°C]
        C_Sn : float
            Sn composition [wt%]
        t_service : float
            Service time [years]
        
        Returns:
        --------
        time_array : np.ndarray
            Time points [years]
        mass_loss : np.ndarray
            Cumulative mass loss [g]
        radius : np.ndarray
            Anode radius evolution [cm]
        """
        # Time discretization
        t_hours = np.linspace(0, t_service * 8760, 1000)  # years to hours
        
        mass_loss = np.zeros_like(t_hours)
        radius = np.ones_like(t_hours) * self.r0
        
        for i in range(1, len(t_hours)):
            dt = t_hours[i] - t_hours[i-1]
            
            # Current density at this time
            i_current = self.model.predict_current(
                t_age, T_age, t_hours[i], C_Sn
            )
            
            # Incremental mass loss via Faraday's law
            dm = (CONST.M_Al * i_current * 1e-3 * 
                  2 * np.pi * radius[i-1] * self.L * dt * 3600) / \
                 (CONST.n * CONST.F)
            
            mass_loss[i] = mass_loss[i-1] + dm
            
            # Update radius (uniform corrosion assumption)
            radius[i] = self.r0 - mass_loss[i] / \
                       (CONST.rho_Al * 2 * np.pi * self.r0 * self.L)
        
        return t_hours / 8760, mass_loss, radius
    
    def predict_lifetime(self, t_age: float, T_age: float, 
                        C_Sn: float, criterion: float = 0.7) -> float:
        """
        Predict service life to failure criterion
        
        Parameters:
        -----------
        t_age : float
            Aging time [hours]
        T_age : float
            Aging temperature [°C]
        C_Sn : float
            Sn composition [wt%]
        criterion : float
            Mass loss fraction at end-of-life (default 0.7)
        
        Returns:
        --------
        lifetime : float
            Predicted service life [years]
        """
        initial_mass = CONST.rho_Al * np.pi * self.r0**2 * self.L
        target_mass_loss = criterion * initial_mass
        
        # Simulate extended service
        t_max = 15  # Maximum 15 years
        t_years, mass_loss, _ = self.integrate_mass_loss(
            t_age, T_age, C_Sn, t_max
        )
        
        # Find time when criterion is exceeded
        idx = np.where(mass_loss >= target_mass_loss)[0]
        
        if len(idx) > 0:
            lifetime = t_years[idx[0]]
        else:
            lifetime = t_max  # Exceeds simulation duration
        
        return lifetime

# ============================================================================
# EXAMPLE USAGE AND VALIDATION
# ============================================================================

def generate_synthetic_data(n_samples: int = 280) -> pd.DataFrame:
    """
    Generate synthetic experimental data for demonstration
    
    Parameters:
    -----------
    n_samples : int
        Number of synthetic measurements
    
    Returns:
    --------
    data : pd.DataFrame
        Synthetic experimental dataset
    """
    np.random.seed(42)
    
    # Experimental design space
    Sn_levels = [0.00, 0.01, 0.05, 0.10]
    T_levels = [25, 130, 160, 190]  # As-cast = 25°C
    t_age_levels = [0, 4, 8, 12]
    t_exp_levels = [48, 96, 144, 192, 240, 288, 336]
    
    data = []
    
    # True parameters (for synthetic data generation)
    true_params = {
        'i0_base': 1.584,
        'k0': 4.82e-5,
        'beta': 2.145,
        'gamma': 8.762
    }
    
    model = CoupledAnodeModel(true_params)
    
    for C_Sn in Sn_levels:
        for T_age in T_levels:
            for t_age in t_age_levels:
                # Skip aging for as-cast
                if T_age == 25 and t_age > 0:
                    continue
                if T_age > 25 and t_age == 0:
                    continue
                    
                for t_exp in t_exp_levels:
                    i_true = model.predict_current(
                        t_age, T_age, t_exp, C_Sn
                    )
                    
                    # Add measurement noise (±10%)
                    noise = np.random.normal(0, 0.1 * i_true)
                    i_exp = max(0.01, i_true + noise)
                    
                    data.append({
                        'C_Sn': C_Sn,
                        'T_age': T_age,
                        't_age': t_age,
                        't_exp': t_exp,
                        'i_exp': i_exp
                    })
    
    return pd.DataFrame(data)

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("Multi-Physics Sacrificial Anode Model - Implementation")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic experimental dataset...")
    data = generate_synthetic_data(n_samples=280)
    print(f"   Generated {len(data)} measurements")
    print(f"   Sn range: {data['C_Sn'].min():.2f} - {data['C_Sn'].max():.2f} wt%")
    print(f"   Temperature range: {data['T_age'].min():.0f} - {data['T_age'].max():.0f} °C")
    
    # Fit model
    print("\n2. Fitting coupled model to experimental data...")
    estimator = ParameterEstimator(data)
    results = estimator.fit()
    
    print(f"   Optimization converged: {results['success']}")
    print(f"   Number of iterations: {results['n_iter']}")
    print(f"\n   Model Performance:")
    print(f"   R² = {results['r2']:.4f}")
    print(f"   RMSE = {results['rmse']:.4f} mA/cm²")
    print(f"   MAPE = {results['mape']:.2f} %")
    
    print(f"\n   Fitted Parameters:")
    for param, value in results['params'].items():
        stderr = results['std_errors'][param]
        print(f"   {param:10s} = {value:12.6e} ± {stderr:.6e}")
    
    # Service life prediction example
    print("\n3. Service life prediction example...")
    model = CoupledAnodeModel(results['params'])
    service = ServiceLifeModel(model, r0=2.0, L=10.0)
    
    conditions = [
        (0.10, 160, 4, "Optimized (0.10 wt% Sn, 160°C/4h)"),
        (0.05, 160, 4, "Moderate (0.05 wt% Sn, 160°C/4h)"),
        (0.00, 130, 8, "Baseline (0.00 wt% Sn, 130°C/8h)")
    ]
    
    for C_Sn, T_age, t_age, description in conditions:
        lifetime = service.predict_lifetime(t_age, T_age, C_Sn)
        print(f"   {description:40s}: {lifetime:.2f} years")
    
    print("\n" + "=" * 70)
    print("Implementation complete. Ready for parameter analysis.")
    print("=" * 70)

if __name__ == "__main__":
    main()