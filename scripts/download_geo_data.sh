#!/bin/bash

# Set variables
output_dir="../data"
github_user="bacinger"
repo_name="f1-circuits"
branch_name="master"
zip_url="https://github.com/${github_user}/${repo_name}/archive/refs/heads/${branch_name}.zip"

# make dir to save data
mkdir -p "$output_dir"

# Download and unzip
curl -L "$zip_url" -o "${output_dir}/${repo_name}.zip"
unzip "${output_dir}/${repo_name}.zip" -d "$output_dir"
rm "${output_dir}/${repo_name}.zip"

echo "Data downloaded and unzipped to $outpud_dir"
