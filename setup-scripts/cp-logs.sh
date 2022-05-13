#!/usr/bin/bash

# to move logs from worker-nodes to main node
# rsync -avz --remove-source-files mtech-2@172.20.46.17:/home/mtech-2/Documents/shiva/logs/ /home/shiva/Documents/bo-prog/distributed-safe-bo/logs/
# rsync -avz --remove-source-files mtech@172.20.46.20:/home/iist/Documents/shiva/logs/ /home/shiva/Documents/bo-prog/distributed-safe-bo/logs/
# rsync -avz --remove-source-files mtech@172.20.46.23:/home/mtech/Documents/shiva/logs/ /home/shiva/Documents/bo-prog/distributed-safe-bo/logs/
rsync -avz --remove-source-files mtech@172.20.46.27:/home/mtech/Documents/bo-prog/distributed-safe-bo/logs/ /home/shiva/Documents/bo-prog/distributed-safe-bo/logs/
rsync -avz --remove-source-files mtech@172.20.46.28:/home/mtech/Documents/bo-prog/distributed-safe-bo/logs/ /home/shiva/Documents/bo-prog/distributed-safe-bo/logs/
rsync -avz --remove-source-files mtech@172.20.46.29:/home/mtech/Documents/bo-prog/distributed-safe-bo/logs/ /home/shiva/Documents/bo-prog/distributed-safe-bo/logs/
rsync -avz --remove-source-files mtech@172.20.46.30:/home/mtech/Documents/bo-prog/distributed-safe-bo/logs/ /home/shiva/Documents/bo-prog/distributed-safe-bo/logs/
dirName=`date "+%d_%B_%H_%M_%S"`_backup
mkdir ./logs/$dirName
mv ./logs/*log* ./logs/$dirName
