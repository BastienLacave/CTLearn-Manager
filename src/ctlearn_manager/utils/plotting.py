"""
Plotting utilities for CTLearn Manager.
Provides a flexible plotting class to handle repetitive plotting logic across different metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Union
from scipy.interpolate import interp1d
import astropy.units as u

from .utils import Cuts, get_color


class PlotManager:
    """
    A manager class to handle plotting with automatic labeling, cuts display,
    and optional relative improvement panels.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Main axes to plot on. If None, a new figure will be created.
    show_relative : bool, optional
        If True, creates a bottom panel showing relative improvement compared to reference.
        Default is False.
    reference_label : str, optional
        Label of the reference curve for relative improvement. Required if show_relative=True.
    figsize : tuple, optional
        Figure size if creating new figure. Default is (10, 6).
    relative_height_ratio : float, optional
        Height ratio for relative panel. Default is 0.25 (25% of main plot height).
    """
    
    def __init__(
        self,
        ax: Optional[plt.Axes] = None,
        show_relative: bool = False,
        reference_label: str = None,
        figsize: Tuple[float, float] = (10, 6),
        relative_height_ratio: float = 0.25,
    ):
        self.show_relative = show_relative
        self.reference_label = reference_label
        self.curves: List[dict] = []
        self.unique_cuts = None
        self.unique_coordinates = None
        
        if show_relative and reference_label is None:
            raise ValueError("reference_label must be specified when show_relative=True")
        
        # Create figure and axes
        if ax is None:
            if show_relative:
                self.fig, (self.ax_main, self.ax_rel) = plt.subplots(
                    2, 1, 
                    figsize=figsize,
                    gridspec_kw={'height_ratios': [1, relative_height_ratio], 'hspace': 0},
                    sharex=True
                )
            else:
                self.fig, self.ax_main = plt.subplots(figsize=figsize)
                self.ax_rel = None
        else:
            self.fig = ax.figure
            self.ax_main = ax
            self.ax_rel = None
            if show_relative:
                raise ValueError("Cannot create relative panel with provided ax. Please create figure without ax parameter.")
    
    def add_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label: str,
        cuts: Optional[Cuts] = None,
        coordinates: Optional[Tuple[u.Quantity, u.Quantity]] = None,
        **plot_kwargs
    ):
        """
        Add a curve to be plotted.
        
        Parameters
        ----------
        x : np.ndarray
            X-axis data.
        y : np.ndarray
            Y-axis data.
        label : str
            Base label for the curve.
        cuts : Cuts, optional
            Cuts object associated with this curve.
        coordinates : tuple of (zenith, azimuth), optional
            Coordinates associated with this curve.
        **plot_kwargs
            Additional keyword arguments for plotting (color, linestyle, marker, etc.).
        """
        self.curves.append({
            'x': x,
            'y': y,
            'label': label,
            'cuts': cuts,
            'coordinates': coordinates,
            'plot_kwargs': plot_kwargs
        })
    
    def _determine_label_strategy(self):
        """
        Determine whether to show cuts/coordinates in labels or as annotation.
        Sets self.unique_cuts and self.unique_coordinates.
        """
        cuts_list = [c['cuts'] for c in self.curves if c['cuts'] is not None]
        coords_list = [c['coordinates'] for c in self.curves if c['coordinates'] is not None]
        
        # Check if all cuts are the same
        if len(cuts_list) > 0:
            cuts_strs = [str(c) for c in cuts_list]
            self.unique_cuts = cuts_list[0] if len(set(cuts_strs)) == 1 else None
        else:
            self.unique_cuts = None
        
        # Check if all coordinates are the same
        if len(coords_list) > 0:
            coords_strs = [f"{c[0].value:.3f}_{c[1].value:.3f}" for c in coords_list]
            self.unique_coordinates = coords_list[0] if len(set(coords_strs)) == 1 else None
        else:
            self.unique_coordinates = None
    
    def _build_full_label(self, curve: dict) -> str:
        """
        Build the full label for a curve based on labeling strategy.
        
        Parameters
        ----------
        curve : dict
            Curve dictionary with label, cuts, and coordinates.
        
        Returns
        -------
        str
            Full label for the curve.
        """
        parts = [curve['label']]
        
        # Add cuts to label if not unique
        if curve['cuts'] is not None and self.unique_cuts is None:
            parts.append(curve['cuts'].get_label())
        
        # Add coordinates to label if not unique
        if curve['coordinates'] is not None and self.unique_coordinates is None:
            zenith, azimuth = curve['coordinates']
            parts.append(f"({zenith.value:.1f}°, {azimuth.value:.1f}°)")
        
        return " | ".join(parts)
    
    def _plot_cuts_annotation(self):
        """Add cuts information as annotation if all curves have the same cuts."""
        if self.unique_cuts is not None:
            self.unique_cuts.plot_cuts_info_plt(self.ax_main)
    
    def _plot_coordinates_annotation(self):
        """Add coordinates information as annotation if all curves have the same coordinates."""
        if self.unique_coordinates is not None:
            from .utils import plot_pointing_on_ax
            zenith, azimuth = self.unique_coordinates
            plot_pointing_on_ax(self.ax_main, zenith, azimuth)
    
    def _compute_relative_improvement(self):
        """
        Compute relative improvement for all curves compared to reference.
        Returns list of (x_interp, relative_improvement) tuples.
        """
        # Find reference curve
        reference_curve = None
        for curve in self.curves:
            full_label = self._build_full_label(curve)
            if self.reference_label in full_label or curve['label'] == self.reference_label:
                reference_curve = curve
                break
        
        if reference_curve is None:
            raise ValueError(f"Reference curve with label '{self.reference_label}' not found")
        
        ref_x = reference_curve['x']
        ref_y = reference_curve['y']
        
        # Remove NaNs from reference
        ref_mask = np.isfinite(ref_y)
        ref_x = ref_x[ref_mask]
        ref_y = ref_y[ref_mask]
        
        if len(ref_x) < 2:
            raise ValueError("Reference curve has too few valid points for interpolation")
        
        # Create interpolation function
        ref_interp = interp1d(ref_x, ref_y, kind='linear', bounds_error=False, fill_value=np.nan)
        
        relative_data = []
        for curve in self.curves:
            x = curve['x']
            y = curve['y']
            
            # Remove NaNs
            mask = np.isfinite(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) == 0:
                relative_data.append((x, np.full_like(x, np.nan)))
                continue
            
            # Interpolate reference to curve's x values
            ref_y_interp = ref_interp(x_clean)
            
            # Compute relative improvement: (y - ref) / ref * 100
            # For sensitivity, lower is better, so improvement = (ref - y) / ref * 100
            relative = (ref_y_interp - y_clean) / ref_y_interp * 100
            
            # Map back to original x array
            rel_full = np.full_like(x, np.nan)
            rel_full[mask] = relative
            
            relative_data.append((x, rel_full))
        
        return relative_data
    
    def plot(
        self,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        xscale: str = "linear",
        yscale: str = "linear",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        show_legend: bool = True,
        legend_kwargs: Optional[dict] = None,
    ):
        """
        Generate the plot with all added curves.
        
        Parameters
        ----------
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        title : str
            Plot title.
        xscale : str
            X-axis scale ('linear', 'log', etc.).
        yscale : str
            Y-axis scale ('linear', 'log', etc.).
        xlim : tuple, optional
            X-axis limits.
        ylim : tuple, optional
            Y-axis limits.
        show_legend : bool
            Whether to show legend. Default is True.
        legend_kwargs : dict, optional
            Additional kwargs for legend.
        """
        self._determine_label_strategy()
        
        # Plot all curves on main axis
        for curve in self.curves:
            full_label = self._build_full_label(curve)
            self.ax_main.plot(
                curve['x'],
                curve['y'],
                label=full_label,
                **curve['plot_kwargs']
            )
        
        # Add annotations
        self._plot_cuts_annotation()
        self._plot_coordinates_annotation()
        
        # Configure main axis
        self.ax_main.set_ylabel(ylabel)
        self.ax_main.set_xscale(xscale)
        self.ax_main.set_yscale(yscale)
        
        if not self.show_relative:
            self.ax_main.set_xlabel(xlabel)
        
        if title:
            self.ax_main.set_title(title)
        
        if xlim:
            self.ax_main.set_xlim(xlim)
        if ylim:
            self.ax_main.set_ylim(ylim)
        
        if show_legend:
            legend_kwargs = legend_kwargs or {}
            self.ax_main.legend(**legend_kwargs)
        
        # Plot relative improvement if requested
        if self.show_relative:
            relative_data = self._compute_relative_improvement()
            
            for curve, (x_rel, y_rel) in zip(self.curves, relative_data):
                # Skip reference curve
                full_label = self._build_full_label(curve)
                if self.reference_label in full_label or curve['label'] == self.reference_label:
                    continue
                
                # Use same styling as main plot
                plot_kwargs = curve['plot_kwargs'].copy()
                plot_kwargs.pop('label', None)  # Remove label
                
                self.ax_rel.plot(x_rel, y_rel, **plot_kwargs)
            
            # Configure relative axis
            self.ax_rel.set_xlabel(xlabel)
            self.ax_rel.set_ylabel("Improvement [%]")
            self.ax_rel.set_xscale(xscale)
            self.ax_rel.axhline(0, color=get_color('on_background'), linestyle='--', linewidth=0.8, alpha=0.5)
            self.ax_rel.grid(True, alpha=0.3)
            
            if xlim:
                self.ax_rel.set_xlim(xlim)
        
        plt.tight_layout()
    
    def save(self, filename: str, **kwargs):
        """
        Save the figure to file.
        
        Parameters
        ----------
        filename : str
            Output filename.
        **kwargs
            Additional arguments for savefig.
        """
        self.fig.savefig(filename, **kwargs)
    
    def show(self):
        """Display the plot."""
        plt.show()
