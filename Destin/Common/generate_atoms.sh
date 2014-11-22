#!/bin/bash

mkdir -p trees
for layers in 2 3 4 5 6 7 ; do

    for trees in 1 3 10 ; do
        ./openCogAtomGenerator $layers $trees > trees/L${layers}_T${trees}.scm
    done
done
