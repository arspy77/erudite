#!/bin/bash
kubectl create -f examples/job1-non-salr.yaml
kubectl create -f examples/job2-non-salr.yaml
kubectl create -f examples/job3-non-salr.yaml
kubectl create -f examples/job4-non-salr.yaml
kubectl get pods -w