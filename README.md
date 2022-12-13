# needle
A machine learning framework project motivated by CMU-10414

# Test:
+ To test the funationality of `backend_edarray`, run

```bash
python3 -m pytest -l -v -k "test_nd_backend"
```

## TODO:
+ support `Conv`
+ fix bugs in the current application(mnist). It doesn't work because we change the backend from numpy to cpu/cuda. 
