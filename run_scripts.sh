#!/bin/bash
for ((i=0;i<10;i++))
do
    # python ./source/bird_sbo.py
    # python ./source/langermann_sbo.py
    python ./source/levy05_sbo.py
    # python ./source/michalewicz_sbo.py
done
# for ((i=0;i<10;i++))
# do
#     python ./source/bird_dbo.py
#     python ./source/langermann_dbo.py
#     python ./source/levy05_dbo.py
#     python ./source/michalewicz_dbo.py
# done
