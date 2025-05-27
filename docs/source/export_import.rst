.. image:: images/CreateModels.png

9. Export and import curves
===========================

There are cases when the users wants to export or import curves from another analysis of the Manager.

Export curves
-------------

To export curves, use `export_to_h5="/path/to/your/exported_curve.h5"` in some of the plot functions to export the curves to an HDF5 file.

Import curves
-------------

To plot the curves from an exported file, use the `import_from_h5="/path/to/your/exported_curve.h5"` and `import_label="Some label"` arguments in the plot functions.


.. image:: images/CreateModelsBottom.png