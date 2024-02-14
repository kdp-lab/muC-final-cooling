#! /usr/bin/bash
mkdir -p ~/Geant4Data
cat geant4files.txt | while read line
do
	echo $line
	curl $line | tar -xz -C ~/Geant4Data
done
