#!/usr/bin/env bash

rm CharacteristicExtractReduction.h5
rm -r extract
cp ~/spectre/build/bin/AnalyticTestCharacteristicExtract .
./AnalyticTestCharacteristicExtract --input-file AnalyticTestBouncingBlackHole.yaml +ppn 2
spectre extract-dat CharacteristicExtractReduction.h5 extract
cd extract
mv SpectreR0*/* .
rm -rf SpectreR0*
rm -rf Cce
