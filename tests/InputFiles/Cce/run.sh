#!/usr/bin/env bash

rm CharacteristicExtractReduction.h5
rm -r extract
cp ~/spectre/build/bin/KleinGordonAnalyticTestCharacteristicExtract .
./KleinGordonAnalyticTestCharacteristicExtract --input-file AnalyticTestBouncingBlackHole.yaml +ppn 2
spectre extract-dat CharacteristicExtractReduction.h5 extract
cd extract
mv SpectreR0*/* .

