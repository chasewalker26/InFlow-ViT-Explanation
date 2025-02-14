<h1>EVALUATION ON ALL MODELS</h1>

`cd InFlow_results/evaluations`

You should have the imagenet ILSVRC2012 validation set of images in a folder to point these tests to.

<h3>Insertion, Deletion, and MAS Insertion and Deletion Tests Without IG Post-Processing</h3>

```
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Naive_Rollout --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Rollout --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Transition_attn_MAP --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Bidirectional_MAP --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function LRP --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function InFlow --model VIT_base_16

python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Naive_Rollout --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Rollout --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Transition_attn_MAP --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Bidirectional_MAP --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function LRP --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function InFlow --model VIT_tiny_16

python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Naive_Rollout --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Rollout --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Transition_attn_MAP --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Bidirectional_MAP --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function LRP --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function InFlow --model VIT_base_32

```


<h3>Insertion, Deletion, and MAS Insertion and Deletion Tests With IG Post-Processing</h3>

```
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Naive_Rollout_IG --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Rollout_IG --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Transition_attn --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Bidirectional --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function LRP_IG --model VIT_base_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function InFlow_IG --model VIT_base_16

python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Naive_Rollout_IG --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Rollout_IG --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Transition_attn --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Bidirectional --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function LRP_IG --model VIT_tiny_16
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function InFlow_IG --model VIT_tiny_16

python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Naive_Rollout_IG --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Rollout_IG --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Transition_attn --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function Bidirectional --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function LRP_IG --model VIT_base_32
python3 evalOnImageNet.py --image_count 5000 --imagenet <path-to-imagenet-validation> --function InFlow_IG --model VIT_base_32

```

<h3>AIC and SIC Tests</h3>

Repeat all above tests as: 


```
python3 evalOnImageNetPIC.py --image_count 1000 --imagenet <path-to-imagenet-validation> --function X --model Y

```


<h3>Runtime Tests</h3>



```
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function Naive_Rollout --model VIT_base_16
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function Rollout --model VIT_base_16
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function Transition_attn --model VIT_base_16
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function Bidirectional --model VIT_base_16
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function LRP --model VIT_base_16
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function InFlow --model VIT_base_16
```

<h3>Evaluation of InFlow Contributions (Table 2)</h3>



```
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function Rollout --model VIT_base_16
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function RAVE --model VIT_base_16
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function RAVE_1 --model VIT_base_16
python3 evalOnImageNet.py --image_count 100 --imagenet <path-to-imagenet-validation> --function RAVE_2 --model VIT_base_16

```

<h3>Converging Connection Contribution Study</h3>


The results are visualized with grapher.ipynb.


```
python3 gatherResidualContribution.py --image_count 10000 --imagenet <path-to-imagenet-validation>

```

<h3>Make Qualitative Results</h3>



```
python3 allAttrCompPDF.py --imagenet <path-to-imagenet-validation>/ --model VIT_base_16
python3 allAttrCompPDF.py --imagenet <path-to-imagenet-validation>/ --model VIT_base_32
python3 allAttrCompPDF.py --imagenet <path-to-imagenet-validation>/ --model VIT_tiny_16

```
