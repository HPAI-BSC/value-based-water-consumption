#!/bin/bash
print_usage() {
  echo "Usage: $0 [-p <path_to_folder>] [-s <seed_number>] [-r <repetitions>]"
}

cd /code

rm -fR /tmp/results
mkdir -p /tmp/results/monthly
mkdir -p /tmp/results/daily
mkdir -p ${RESULTS_PATH}

# Default values
path_to_folder="/tmp/results/"
seed_number=""
repetitions="1"

# Parse the options and arguments using getopt
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p | --path)
      path_to_folder="$2"
      shift 2
      ;;
    -s | --seed)
      seed_number="$2"
      shift 2
      ;;
    -r | --repetitions)
      repetitions="$2"
      shift 2
      ;;
    -h)
      print_usage
      exit 1
      ;;
    *)
      shift 1
      ;;
  esac
done

xmlstarlet ed --inplace -u "experiments/experiment/enumeratedValueSet[@variable='path-to-results']/value/@value" -v "\"${path_to_folder}\"" /code/setup.xml
xmlstarlet ed --inplace -u "experiments/experiment/@repetitions" -v "${repetitions}" /code/setup.xml

if [ -n "$seed_number" ]; then
  xmlstarlet ed --inplace -u "experiments/experiment/enumeratedValueSet[@variable='seed-value']/value/@value" -v "${seed_number}" /code/setup.xml
  xmlstarlet ed --inplace -u "experiments/experiment/enumeratedValueSet[@variable='gen-seed']/value/@value" -v "false" /code/setup.xml
else
  xmlstarlet ed --inplace -u "experiments/experiment/enumeratedValueSet[@variable='gen-seed']/value/@value" -v "true" /code/setup.xml
fi

echo "Running experiments, consolidated setup:"
cat /code/setup.xml

echo "Starting simulations..."
time /netlogo/netlogo-headless.sh --model /code/NLModel.nlogo --setup-file /code/setup.xml --experiment experiment -D-Xmx94208m
#java -Xmx2048m -Dfile.encoding=UTF-8 -classpath NetLogo.jar org.nlogo.headless.Main --model /code/NLModel.nlogo --setup-file /code/setup.xml --experiment experiment
#
# PYTHONPATH=/code/python-scripts python3 python-scripts/consolidate_results.py

# echo "Compressing results..."
# gzip -c ${RESULTS_PATH}/all.csv > ${RESULTS_PATH}/all.zip
# rm ${RESULTS_PATH}/all.csv
