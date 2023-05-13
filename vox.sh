#!/bin/bash
for dir in benches/*; do
    cd $dir
    rm *.binvox
    ../../binvox -d 64 model_watertight.obj
    pwd
    cd ../..

done