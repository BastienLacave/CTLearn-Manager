# PlotManager Usage Examples

The `PlotManager` class provides a unified interface for creating plots with automatic handling of cuts and coordinate labeling, as well as optional relative improvement panels.

## Basic Usage

```python
from ctlearn_manager.utils import PlotManager, Cuts, CutType
import numpy as np
import astropy.units as u

# Create a PlotManager instance
plot_manager = PlotManager()

# Add curves with different cuts
x = np.logspace(-2, 1, 20)
y1 = x ** -2.5
y2 = x ** -2.3

cut1 = Cuts(cut_type=CutType.GLOBAL, gammaness_cut=0.7)
cut2 = Cuts(cut_type=CutType.GLOBAL, gammaness_cut=0.9)

plot_manager.add_curve(x, y1, label="Model A", cuts=cut1, marker='o', ls='--')
plot_manager.add_curve(x, y2, label="Model B", cuts=cut2, marker='s', ls='-')

# Generate the plot
plot_manager.plot(
    xlabel="Energy [TeV]",
    ylabel="Sensitivity [% Crab]",
    title="Differential Sensitivity",
    xscale="log",
    yscale="log"
)

# Save or show
plot_manager.save("sensitivity.png")
# or
plot_manager.show()
```

## Features

### 1. Automatic Cut Labeling

If all curves have the **same cuts**, the cuts are displayed as an annotation in the bottom-right corner:

```python
plot_manager = PlotManager()

# All curves use the same cut
same_cut = Cuts(cut_type=CutType.GLOBAL, gammaness_cut=0.7)

plot_manager.add_curve(x, y1, label="Model A", cuts=same_cut)
plot_manager.add_curve(x, y2, label="Model B", cuts=same_cut)

# The cuts will appear as annotation, not in labels
plot_manager.plot(xlabel="Energy [TeV]", ylabel="Sensitivity")
```

If curves have **different cuts**, each cut is appended to its curve's label:

```python
plot_manager = PlotManager()

cut1 = Cuts(cut_type=CutType.GLOBAL, gammaness_cut=0.7)
cut2 = Cuts(cut_type=CutType.GLOBAL, gammaness_cut=0.9)

plot_manager.add_curve(x, y1, label="Model A", cuts=cut1)  # Label: "Model A | G/H cut: 0.7"
plot_manager.add_curve(x, y2, label="Model B", cuts=cut2)  # Label: "Model B | G/H cut: 0.9"

plot_manager.plot(xlabel="Energy [TeV]", ylabel="Sensitivity")
```

### 2. Coordinate Annotation

Similarly, if all curves have the same coordinates (zenith, azimuth), they appear as an annotation:

```python
plot_manager = PlotManager()

coords = (10.0 * u.deg, 248.0 * u.deg)  # (zenith, azimuth)

plot_manager.add_curve(x, y1, label="Model A", coordinates=coords)
plot_manager.add_curve(x, y2, label="Model B", coordinates=coords)

# Coordinates will appear as annotation in bottom-left
plot_manager.plot(xlabel="Energy [TeV]", ylabel="Sensitivity")
```

### 3. Relative Improvement Panel

Show relative improvement compared to a reference curve:

```python
plot_manager = PlotManager(
    show_relative=True,
    reference_label="Model A",  # Reference for comparison
    figsize=(10, 8),
    relative_height_ratio=0.25
)

plot_manager.add_curve(x, y1, label="Model A", marker='o')
plot_manager.add_curve(x, y2, label="Model B", marker='s')
plot_manager.add_curve(x, y3, label="Model C", marker='^')

# This creates:
# - Top panel: main plot with all curves
# - Bottom panel: relative improvement of Model B and C compared to Model A
#   Improvement % = (y_ref - y_curve) / y_ref * 100
plot_manager.plot(
    xlabel="Energy [TeV]",
    ylabel="Sensitivity [% Crab]",
    title="Sensitivity Comparison"
)
```

The bottom panel:
- Shows improvement as percentage
- Automatically interpolates reference curve to match x-values
- Shares x-axis with top panel (no gap between panels)
- Omits the reference curve itself

### 4. Using with Existing Axes

You can pass an existing axis to plot on:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

plot_manager = PlotManager(ax=ax)
plot_manager.add_curve(x, y, label="My Data")
plot_manager.plot(xlabel="X", ylabel="Y")

plt.show()
```

**Note:** You cannot create a relative improvement panel when providing your own axes.

## Refactoring Existing Plot Functions

Here's how to refactor an existing plotting function to use `PlotManager`:

### Before (original code):

```python
def plot_sensitivity(self, ax=None, label="CTLearn", output_file=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if len(self.cuts) == 1:
        self.cuts[0].plot_cuts_info_plt(ax)
    
    for i, cut in enumerate(self.cuts):
        E = (self.E_bins[i][:-1] + self.E_bins[i][1:]) / 2
        sensitivity = self.compute_sensitivity(cut)
        
        if len(self.cuts) > 1:
            ax.plot(E, sensitivity, marker='o', label=cut.get_label())
        else:
            ax.plot(E, sensitivity, marker='o', label=label)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [TeV]")
    ax.set_ylabel("Sensitivity [% Crab]")
    ax.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
```

### After (using PlotManager):

```python
def plot_sensitivity(
    self, 
    ax=None, 
    label="CTLearn", 
    output_file=None,
    show_relative=False,
    reference_label=None
):
    plot_manager = PlotManager(
        ax=ax,
        show_relative=show_relative,
        reference_label=reference_label
    )
    
    for i, cut in enumerate(self.cuts):
        E = (self.E_bins[i][:-1] + self.E_bins[i][1:]) / 2
        sensitivity = self.compute_sensitivity(cut)
        
        # Base label only, PlotManager handles cut labeling
        base_label = cut.get_label() if len(self.cuts) > 1 else label
        
        plot_manager.add_curve(
            x=E.value,
            y=sensitivity,
            label=base_label,
            cuts=cut,
            marker='o',
            ls='--'
        )
    
    plot_manager.plot(
        xlabel="Energy [TeV]",
        ylabel="Sensitivity [% Crab]",
        title="Differential Sensitivity",
        xscale="log",
        yscale="log"
    )
    
    if output_file:
        plot_manager.save(output_file)
    else:
        plot_manager.show()
```

## Benefits

1. **Reduced Code Duplication**: Common plotting logic is centralized
2. **Consistent Styling**: All plots follow the same conventions
3. **Automatic Labeling**: Smart handling of cuts and coordinates
4. **Easy Comparisons**: Built-in relative improvement panels
5. **Flexible**: Works with or without provided axes
6. **Clean Code**: Separates data preparation from presentation logic
