# needle
A machine learning framework project motivated by CMU-10414

## Test:
+  `backend_ndarray` test

```bash
python3 -m pytest -l -v -k "test_nd_backend"
```

+ `dataset` test
```bash
python3 -m pytest -v -k "test_flip_horizontal"
python3 -m pytest -v -k "test_random_crop"
python3 -m pytest -v -k "test_mnist_dataset"
python3 -m pytest -v -k "test_dataloader_mnist"
python3 -m pytest -v -k "test_dataloader_ndarray"

# run all the tests in one command
python3 -m pytest -l -v -k "test_data"
```

+ `conv` test
```bash
python3 -m pytest -l -v -k "test_conv"
```


## TODO:

+ `nn.LayerNorm1d` can't work correctly now, because the new shape in `broadcast_to` may have the diifferent dimension with the old shape. 