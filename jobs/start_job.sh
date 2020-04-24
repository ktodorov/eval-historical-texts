#!/bin/bash

filepath=$1
shift

str="ALL"

for arg in $@
do
    str="$str,$arg"
done

command="sbatch --export=$str $filepath"
echo "Executing '$command'"

sbatch --export=$str $filepath

echo "Command executed"
