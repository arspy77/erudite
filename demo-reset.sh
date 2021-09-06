#!/bin/bash
kubectl delete tfjob job1-salr
kubectl delete tfjob job2-salr
kubectl delete tfjob job3-salr
kubectl delete tfjob job4-salr
kubectl delete tfjob job1-non-salr
kubectl delete tfjob job2-non-salr
kubectl delete tfjob job3-non-salr
kubectl delete tfjob job4-non-salr