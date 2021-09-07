#!/bin/bash
kubectl create -f examples/job1-salr.yaml
kubectl create -f examples/job2-salr.yaml
kubectl create -f examples/job3-salr.yaml
kubectl create -f examples/job4-salr.yaml
kubectl get pods -w