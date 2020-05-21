CHECKPOINT=$1

#python eval.py --dataset cifar10 --arch wideresnet --out cifar10@4000 --checkpoint ${CHECKPOINT}
python eval.py --dataset cifar10 --arch wideresnet --checkpoint ${CHECKPOINT}
