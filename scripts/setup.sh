#!/bin/bash

set -euC

main(){
  git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
  git clone https://github.com/LeviBorodenko/motionblur motionblur
  chmod u+x scripts/run_sampling.sh
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main
fi
