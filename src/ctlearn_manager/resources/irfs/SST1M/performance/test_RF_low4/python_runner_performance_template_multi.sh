#!/bin/bash

INPUT_FILE_P=$1
INPUT_FILE_G=$2
CONFIG=$3
OUTPATH=$4
G_CUT=$5
VERSION=$6

source /home/jurysek/.bashrc
conda activate sst1m-$VERSION

echo $TMPDIR
tmpdir=$(echo $TMPDIR)
cp $INPUT_FILE_P $tmpdir
cp $INPUT_FILE_G $tmpdir
cp $CONFIG $tmpdir
inputfile_p_no_path=${INPUT_FILE_P##*/}
inputfile_g_no_path=${INPUT_FILE_G##*/}
config_no_path=${CONFIG##*/} 

sst1mpipe_mc_performance -fp $tmpdir/$inputfile_p_no_path -fg $tmpdir/$inputfile_g_no_path -c $tmpdir/$config_no_path -o $tmpdir --save-fig --save-hdf --gammaness-cuts=$G_CUT  --sensitivity --rf-performance

mv $tmpdir/*tel_001.h5 $OUTPATH
mv $tmpdir/*tel_002.h5 $OUTPATH
mv $tmpdir/*stereo.h5 $OUTPATH
mv $tmpdir/*.png $OUTPATH
mv $tmpdir/*log $OUTPATH
mv $tmpdir/roc_bins $OUTPATH
