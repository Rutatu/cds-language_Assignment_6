#!/usr/bin/env bash

VENVNAME=GoT
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME