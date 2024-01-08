# EventSkeletonGeneration

## Experiment code and data for: "A Diffusion Model for Event Skeleton Generation"

### Step 1: Prepare the Dataset

Option 1: Download data from the provided [Google drive link](https://drive.google.com/file/d/1olcrCKgsHJSy06DlHRMPnjAtzdd0UbkE/view?usp=sharing).

Option 2: Process the data yourself

- Download the raw data from [temporal-graph-schema](https://github.com/limanling/temporal-graph-schema).
- Process the data according to the instructions found in [Event Schema Induction with Double Graph Autoencoders](https://aclanthology.org/attachments/2022.naacl-main.147.software.zip).

### Step 2: Train the Model
- Execute the `run.sh` script to train the model. Training outcomes are stored in `./*.log` files. The `run.sh` script will perform training five times for each dataset to compute the average results.

### Step 3: Compile Results
- Run `stat_log.py` to print the summary statistics of training results to the console.

### Note
Even though we use the method of averaging over five runs, the training results are still quite unstable. Future work may consider improving evaluation criteria to enhance the stability of evaluation.

An intriguing analogy is that when you average the faces of the general population, you typically end up with an exceptionally beautiful face. In a similar vein, a well-constructed event schema graph may bear a striking resemblance to various individual event instance graphs.

## Reference
```
@inproceedings{zhu-etal-2023-diffusion,
    title = "A Diffusion Model for Event Skeleton Generation",
    author = "Zhu, Fangqi and Zhang, Lin and Gao, Jun and Qin, Bing and Xu, Ruifeng and Yang, Haiqin",
    editor = "Rogers, Anna and Boyd-Graber, Jordan and Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.800",
    doi = "10.18653/v1/2023.findings-acl.800",
    pages = "12630--12641",
}
```
