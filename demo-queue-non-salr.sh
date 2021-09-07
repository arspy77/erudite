#!/bin/bash
kubectl create -f examples/job1-queue-non-salr.yaml
kubectl create -f examples/job2-queue-non-salr.yaml
kubectl create -f examples/job3-queue-non-salr.yaml
kubectl create -f examples/job4-queue-non-salr.yaml
kubectl create -f examples/job5-queue-non-salr.yaml
kubectl create -f examples/job6-queue-non-salr.yaml
kubectl create -f examples/job7-queue-non-salr.yaml
kubectl create -f examples/job8-queue-non-salr.yaml
kubectl create -f examples/job9-queue-non-salr.yaml
kubectl create -f examples/job10-queue-non-salr.yaml
kubectl create -f examples/job11-queue-non-salr.yaml
kubectl create -f examples/job12-queue-non-salr.yaml
kubectl get pods -w