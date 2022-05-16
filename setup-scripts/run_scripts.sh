#!/bin/bash

for ((i=0;i<5;i++))
do
    python ./results/levy05_function/levy05_hyperspaces.py
    sleep 5
done
for ((i=0;i<5;i++))
do
    python ./results/levy05_function/levy05_hyperspaces_ovr.py
    sleep 5
done
for ((i=0;i<5;i++))
do
    python ./results/levy05_function/levy05_sbo.py
    sleep 5
done
