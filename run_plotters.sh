#!/bin/bash

find . -type d -name "plot_conf" | while read -r conf_dir; do
    echo "Processing YAML files in: $conf_dir"
    for yaml_file in "$conf_dir"/*.yaml "$conf_dir"/*.yml; do
        if [ -f "$yaml_file" ]; then
            echo "Running plotter on: $yaml_file"
            python plotter.py "$yaml_file"
        fi
    done
done 