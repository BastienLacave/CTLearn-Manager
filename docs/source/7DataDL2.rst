7. Real data DL2 analysis
=========================

Predicting real data is done with a ``TriModelManager``, or a ``TriModelCollection`` of multiple models can be applied to a larger set of data, for example if each model correspond to a different pointing in the sky, the relevant one will predict each file.

DL2DataProcessor
----------------

The ``DL2DataProcessor`` is a tool for DL2 data analysis, it can produce sensitivity curves, PSF, background discrimination capability of your model, theta square plots, sky maps with on and off regions. The Processor needs a TriModelManager that corresponds an equivalent model used to predict the data, meaning one of the Collection used for the dataset. This is to ensure the telescopes are the same.
The first setp is to process the DL2 data, using slurm is recommended, to extract excess for a range of cuts, as well as compute the sky coordinates of all events. They are stored and pickled in a directory of choice. This is needed only once per file.
Then plot !

RFCounterpart
-------------

Used on the LST cluster, ``RFCounterpart`` will process the same runs as passed by the ``DL2DataProcessor``, they are the same class so the same features can be extracted.

Combinator2000
--------------

``Combinator2000`` enables to take energy, direction and type from different sources, CTLearn models or RF, and extract the same features.

WhoIsBetter
-----------
``WhoIsBetter`` is a class that takes Processors or ``RFCounterparts`` or Combinator2000s and overlay the curves and the same plots.
