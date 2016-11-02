#!/bin/bash

VIEWSYN="../../viewsyn/ViewSyn"

echo "Running ViewSyn with solid bg to reproject"
$VIEWSYN deep_test.cfg
echo "Done!"