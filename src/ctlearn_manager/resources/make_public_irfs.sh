#/usr/env bash

ctapipe-optimize-event-selection -c public-conf.yml --gamma-file ./Prod5b_public_MC/gamma_final.dl2.h5 --proton-file ./Prod5b_public_MC/proton_final.dl2.h5 -v --point-like --output ./Tool_IRF/cuts_public.fits  --overwrite True -EventSelectionOptimizer.optimization_algorithm=PercentileCuts

ctapipe-compute-irf -c match-ed-2.yml --IrfTool.cuts_file ./Tool_IRF/cuts_public.fits --gamma-file ./Prod5b_public_MC/gamma_final.dl2.h5 --proton-file ./Prod5b_public_MC/proton_final.dl2.h5  -v --do-background --point-like --output ./Tool_IRF/public_mc_point-like.fits --benchmark-output ./Tool_IRF/public_mc_benchmark.fits
