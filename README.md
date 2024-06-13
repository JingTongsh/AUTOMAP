# AUTOMAP Replication

Code is largely based on the [official repository](https://github.com/MattRosenLab/AUTOMAP.git).

The only things we changed are:

* training and evaluation with our proposed dataset [MORE](https://more-med.github.io/);
* converting data format to `.mat` for AUTOMAP with `convert_data.py`;
* calculating K-space from images rather than reading from files;
* evaluating computational resources with `infer_256.py`.

Results can be replicated with the following steps.

1. Download our [MORE](https://more-med.github.io/) dataset, put the `MRI` directory in the current directorcy, or create a symbol link

```shell
ln -s /path/to/MORE/MRI .
```

2. Convert data to `.mat` format, automatically creating `data-fft` directory.

```shell
python convert_data.py
```

3. Train AUTOMAP model, with auto ckpt in `experiments`

```shell
python automap_main_train.py -c configs/train_64x64_ourdata.json
# or the config file for another scale
```

4. Specify the checkpoint path in the corresponding inference conig file `configs/inference_64x64_ourdata.json` (or that of another scale).

5. Evaluate AUTOMAP model, automatically saving metrics and visualization results to the directory specified in the inference conig file

```shell
python evaluate.py -c configs/inference_64x64_ourdata.json
```
