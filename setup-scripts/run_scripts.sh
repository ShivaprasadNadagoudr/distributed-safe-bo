#!/bin/bash

###########################################################################

for ((i=0;i<20;i++))
do
    python ./results/langermann_function/langermann_hyperspaces.py
done
for ((i=0;i<20;i++))
do
    python ./results/langermann_function/langermann_hyperspaces_ovr.py
done

###########################################################################

for ((i=0;i<20;i++))
do
    python ./results/levy05_function/levy05_hyperspaces.py
done
for ((i=0;i<20;i++))
do
    python ./results/levy05_function/levy05_hyperspaces_ovr.py
done

###########################################################################

for ((i=0;i<20;i++))
do
    python ./results/michalewicz_function/michalewicz_hyperspaces.py
done
for ((i=0;i<20;i++))
do
    python ./results/michalewicz_function/michalewicz_hyperspaces_ovr.py
done

###########################################################################

for ((i=0;i<20;i++))
do
    python ./results/langermann_function/langermann_sbo.py
done

for ((i=0;i<20;i++))
do
    python ./results/levy05_function/levy05_sbo.py
done

for ((i=0;i<20;i++))
do
    python ./results/michalewicz_function/michalewicz_sbo.py
done
