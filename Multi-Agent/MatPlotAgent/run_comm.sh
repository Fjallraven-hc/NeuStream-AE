#!/bin/bash
if [ -d "log" ]; then
    rm -rf log/*
else
    mkdir log
fi
python connect_comm.py