#!/bin/bash
# Run every example on the doraemon final sample at S3DF (develop container).
#   bash examples/doraemon_final/run_examples.sh [wire|pixel]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RO=${1:-wire}
singularity exec -B /sdf,/lscratch /sdf/group/neutrino/images/develop.sif bash -c "
export PYTHONPATH=$REPO/src:$HERE
cd $HERE
for ex in step_labl hits_labl step_hits_labl no_labl; do
  echo \"===== \$ex ($RO)\"
  python \$ex.py $RO
done"
